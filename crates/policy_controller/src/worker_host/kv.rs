//! The worker's KV port over the store's index. The store's cursor reads a
//! batch's header, and an admitted batch is applied inside the row's lane
//! by the worker crate's apply rules, the ones every index of cache events
//! is written by.
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
//! The store empties a rank in its own turn: a resync, a restart or a gap
//! its cursor reads, the worker reported gone. The lane says a value that
//! moves with each emptying, and a batch that reads another value than the
//! row's last one resets the row's state first. The table's lock is let go
//! before a row's state is locked, and a row's state is locked inside its
//! lane's turn alone.
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
//! moves between two streams in one step, and the store clears it in
//! that step, so the clear takes nothing a later stream wrote. A stream
//! that leaves reaches only the rows it still writes. It clears them in
//! the store outside that lock, which every row's batch takes, and each
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
//! A batch the store takes that carries a store names the model's block
//! size: the row's block of record, the widest block its stores carried
//! since its last clear. The engine's word only rises: an engine also
//! publishes prefix entries narrower than its block, and a request hashed
//! at their width would match no whole block. The size of record is put to
//! the engine's word wherever the two differ. So the first stored block
//! replaces a size a registration named, and the next one puts back a size
//! a registration raised behind it. A worker that states no size needs no
//! other word. The adapter keeps the model each open row streams for and
//! the engine's word for each model. The store keeps the size of record.
//!
//! # The series
//!
//! A batch the store takes is applied and accounted for by the worker
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
//! - An adapter made with no holder writes every batch to its stream's row.
//! - A feed's own row laid again in place as rank 0 of another engine is
//!   written by its old stream until that stream ends. Its end clears
//!   every row it wrote.
//! - What a row's old stream had in flight as the row moved, a batch or a
//!   reset, can land behind the new stream's clear.
//! - What the store's cursor decides by itself is in no series: a gap kept
//!   or cleared, the batches missed, the degraded ranks, and the reason it
//!   emptied a rank ahead of a batch.
//! - The index-blocks gauge moves at a batch: a rank the store emptied
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

use super::Objects;
use crate::{
    engine::view::WorkerId,
    store::{
        self, ApplyError, ChainBlockMap, KvEvent, LaneWriter, SequenceHash, StoredBlock,
        WorkerGroups,
    },
};

/// The worker crate's KV event port over a store's KV event sink.
pub struct KvAdapter<S: ?Sized> {
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
    store: Arc<S>,
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
    /// the store yet: no stream takes the row until it has.
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
    /// batch that reads another resets the state first: the store emptied
    /// the rank in between.
    emptied: Option<u64>,
    /// The blocks the row's map held behind its last batch.
    blocks: usize,
}

/// The row's reverse map inside the rules' state. The map is the lane's,
/// since the store empties a rank by it. So this holds it for a batch
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
/// map back, also behind rules that unwound: the store then empties the
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

impl<S: ?Sized> KvAdapter<S> {
    /// An adapter that writes `store`, every batch to its stream's row.
    pub fn new(store: Arc<S>) -> Self {
        Self::over(store, None)
    }

    fn over(store: Arc<S>, ranks: Option<Arc<dyn Ranks>>) -> Self {
        Self {
            rows: Mutex::new(Rows::default()),
            sizes: Mutex::new(Sizes::default()),
            urls: Mutex::new(HashMap::new()),
            ranks,
            store,
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
}

impl<S> KvAdapter<S>
where
    S: store::KvEventSink<Batch = [KvEvent]> + ?Sized,
{
    /// The state of `row` for a batch on `id`'s stream, and whether the
    /// table keeps it: the stream's own row that no entry is kept of gets a
    /// state the table takes once the store took a batch ([`Self::keep`]).
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
    /// store kept of it was decided by facts that are gone, or by another
    /// stream's batches. The stream that wrote it reaches it no longer:
    /// the row changes lists, and the store clears it, in one hold of the
    /// lock. The clear is of one row, and of nothing where the store never
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
        self.store.removed(row);
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
    /// store kept of its stream was decided by facts that are gone. The
    /// store drops the row's lane at whatever rank it keeps it, and a row
    /// with none loses nothing. A row another stream wrote, as a rank of
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
            self.store.removed(row);
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
        // Held through the store's calls: two rows of one model learn in turn.
        let mut sizes = self.sizes.lock();
        let Sizes { models, named } = &mut *sizes;
        let Some(model) = models.get(&id) else {
            return;
        };
        let last = named.get(model).copied();
        let word = last.map_or(size, |last| last.max(size));
        let of_record = self.store.block_size(model);
        if of_record != Some(word) {
            // The store's setter only raises, so a wider size that no stored
            // block named goes first. A request hashed between the two calls
            // reads no size and states no hash.
            if of_record.is_some_and(|of_record| of_record > word) {
                self.store.forget_block_size(model);
            }
            self.store.set_block_size(model, word);
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

impl<S> KvEventSink for KvAdapter<S>
where
    S: store::KvEventSink<Batch = [KvEvent]> + ?Sized + 'static,
{
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
        let facts = store::BatchFacts {
            sequence: facts.sequence,
            clears: facts.clears,
            snapshot: facts.snapshot.map(|chunk| store::SnapshotChunk {
                unknown_before: chunk.unknown_before,
            }),
        };
        // A batch that carries a store names the model's block size.
        let stores = batch
            .events
            .iter()
            .any(|event| matches!(event.data, Some(Data::Stored(_))));
        let mut record = None;
        let mut took = None;
        // The cursor reads the header alone. The body runs where it admits
        // the batch: inside the lane's turn, with the table's lock let go.
        let outcome = self.store.admit_through(row, rank, facts, &mut |lane| {
            let mut applied = applied.lock();
            let Applied {
                state,
                emptied,
                blocks,
            } = &mut *applied;
            if *emptied != Some(lane.emptied()) {
                // The store emptied the rank since the row's last batch, or
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
            store::BatchOutcome::Applied => {
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
                BatchOutcome::Applied
            }
            store::BatchOutcome::Skipped => {
                // A batch behind its lane's cursor. A row with no lane
                // skips every batch and counts none.
                let behind = || self.store.cursor(row, rank).is_some();
                if let Some(url) = url.as_ref().filter(|_| behind()) {
                    Metrics::record_kv_event_batch(url, "stale");
                }
                BatchOutcome::Skipped
            }
            store::BatchOutcome::Gap { expected, received } => {
                // The collector asks the publisher to replay from here.
                if let Some(url) = &url {
                    Metrics::record_kv_event_gap(url, "replay_requested", 0);
                }
                BatchOutcome::Gap { expected, received }
            }
        }
    }

    fn reconnected(&self, id: WorkerId) {
        for row in self.written_by(id) {
            self.store.reconnected(row);
        }
    }

    fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        if let Some(url) = self.urls.lock().get(&id) {
            Metrics::record_kv_event_resync(url, reason.as_str());
        }
        let reason = match reason {
            ResyncReason::OutOfRange => store::ResyncReason::OutOfRange,
            ResyncReason::DataLoss => store::ResyncReason::DataLoss,
            ResyncReason::SnapshotAbandoned => store::ResyncReason::SnapshotAbandoned,
            ResyncReason::PublisherRestart => store::ResyncReason::PublisherRestart,
            ResyncReason::GapCleared => store::ResyncReason::GapCleared,
            ResyncReason::Snapshot => store::ResyncReason::Snapshot,
        };
        match rank {
            // One rank's reset is its own row's, and a rank with no row
            // holds nothing to reset.
            Some(rank) => {
                if let Some(row) = self.routed(id, rank) {
                    self.store.resync(row, Some(rank), reason);
                }
            }
            // The stream's reset reaches every row it writes, its own
            // first.
            None => {
                for row in self.written_by(id) {
                    self.store.resync(row, None, reason);
                }
            }
        }
    }

    fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
        let row = self.routed(id, rank)?;
        self.store.cursor(row, rank).map(|cursor| KvCursor {
            applied: cursor.applied,
            replay_pending: cursor.replay_pending,
            degraded: cursor.degraded,
        })
    }

    fn removed(&self, id: WorkerId) {
        self.sizes.lock().models.remove(&id);
        self.urls.lock().remove(&id);
        // What the stream still writes goes with it, since no stream feeds
        // it now: the rows of other ranks, and its own unless another
        // stream took it. Each is marked in the one hold that reads the
        // list, cleared in the store outside the lock, and forgotten. A
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
            self.store.removed(row);
        }
        let mut rows = self.rows.lock();
        for row in &leaving {
            rows.written.remove(row);
        }
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.store.block_size(model)
    }

    fn set_block_size(&self, model: &str, size: u32) {
        self.store.set_block_size(model, size);
    }
}

/// Why a subscription got no feed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// The holder keeps no row for the worker's object: it laid none, or it
    /// has laid another object in its place.
    NoRow,
    /// The worker's row is a rank above 0 of a data-parallel worker.
    UpperRank,
}

/// How many subscriptions a host gave no feed, by the reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkippedSubscriptions {
    /// The holder kept no row for the worker's object.
    pub no_row: u64,
    /// The worker's row was a rank above 0.
    pub upper_rank: u64,
}

/// The KV collector's host over the store an events subscription feeds.
pub struct KvHost<G, S: ?Sized> {
    objects: Arc<Objects<G>>,
    sink: Arc<KvAdapter<S>>,
    no_row: AtomicU64,
    upper_rank: AtomicU64,
}

impl<G: Generation, S: ?Sized> KvHost<G, S> {
    /// A host whose subscriptions feed `store`, each under the row
    /// `objects` holds for its worker. A batch of another rank of the
    /// worker's engine is written to the row `objects` holds for that rank.
    pub fn new(objects: Arc<Objects<G>>, store: Arc<S>) -> Self {
        let ranks: Arc<dyn Ranks> = Arc::clone(&objects) as _;
        Self {
            objects,
            sink: Arc::new(KvAdapter::over(store, Some(ranks))),
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

impl<G, S: ?Sized> KvHost<G, S> {
    /// How many subscriptions got no feed so far, by the reason.
    pub fn skipped_subscriptions(&self) -> SkippedSubscriptions {
        SkippedSubscriptions {
            no_row: self.no_row.load(Ordering::Relaxed),
            upper_rank: self.upper_rank.load(Ordering::Relaxed),
        }
    }
}

impl<G, S> KvEventHost for KvHost<G, S>
where
    G: Generation,
    S: store::KvEventSink<Batch = [KvEvent]> + ?Sized + 'static,
{
    type Sink = KvAdapter<S>;
    type Ticket = Result<WorkerId, SkipReason>;

    fn enroll(&self, worker: &Arc<dyn Worker>, model: &str) -> Self::Ticket {
        // A size the store's width cannot hold is not a size of record.
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
        self.sink.store.forget_block_size(model);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{OnceLock, Weak};

    use smg_worker::{
        common_proto::{
            KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheCleared, KvCacheEvent,
            KvCacheLocality, KvCacheTier,
        },
        identity::builder::BasicWorkerBuilder,
        kv_events::port::SnapshotChunk,
        registry::WorkerRegistry,
    };

    use super::*;
    use crate::{
        store::{KvEventSink as _, MemoryStore, Signals, WorkerFacts, WorkersStore},
        worker_host::Members,
    };

    const MODEL: &str = "m";
    const URL: &str = "http://a:8000";
    const RANK: u16 = 1;
    const BLOCK_SIZE: u32 = 32;

    type Store = dyn store::KvEventSink<Batch = [KvEvent]>;

    struct Fixture {
        store: Arc<MemoryStore>,
        sink: Arc<KvAdapter<Store>>,
        row: WorkerId,
    }

    impl Fixture {
        /// A row at rank 1 of a worker of two: the adapter's cases name the
        /// rank a batch is forwarded under.
        fn new() -> Self {
            Self::at(RANK, RANK + 1, Some(BLOCK_SIZE))
        }

        /// A worker of one rank: the row a host gives a feed.
        fn single(kv_block_size: Option<u32>) -> Self {
            Self::at(0, 1, kv_block_size)
        }

        fn at(dp_rank: u16, dp_size: u16, kv_block_size: Option<u32>) -> Self {
            let store = Arc::new(MemoryStore::new());
            let row = store
                .upsert(WorkerFacts {
                    key: URL.into(),
                    engine: URL.into(),
                    incarnation: 1,
                    models: vec![MODEL.into()],
                    dp_rank,
                    dp_size,
                    kv_block_size,
                    signals: Signals {
                        events: true,
                        ..Signals::default()
                    },
                    ..WorkerFacts::default()
                })
                .id();
            let sink = Arc::new(KvAdapter::new(Arc::clone(&store) as Arc<Store>));
            Self { store, sink, row }
        }

        fn applied(&self, sequence: u64, batch: &KvEventBatch) {
            assert_eq!(
                self.sink
                    .admit(self.row, RANK, facts_of(sequence, batch), batch),
                BatchOutcome::Applied
            );
        }

        fn blocks(&self) -> usize {
            self.store.indexed_blocks(self.row, RANK)
        }

        fn host(&self) -> KvHost<Members, Store> {
            let objects = Arc::new(Objects::new(Arc::new(WorkerRegistry::new())));
            objects.follow(self.store.as_ref());
            KvHost::new(objects, Arc::clone(&self.store) as Arc<Store>)
        }
    }

    fn facts(sequence: u64) -> BatchFacts {
        BatchFacts {
            sequence,
            ..BatchFacts::default()
        }
    }

    /// Whether the adapter keeps `row` as its own stream's, begun from
    /// nothing: no batch was applied under its state.
    fn begun(sink: &KvAdapter<Store>, row: WorkerId) -> bool {
        let rows = sink.rows.lock();
        rows.written.get(&row).is_some_and(|written| {
            written.stream == row && !written.leaving && written.applied.lock().emptied.is_none()
        })
    }

    /// The header's facts of `batch` at `sequence`, by the collector's rule:
    /// a clear no name owns is the header's clear, and a clear a name owns
    /// is not.
    fn facts_of(sequence: u64, batch: &KvEventBatch) -> BatchFacts {
        let clears = batch.events.iter().any(|event| {
            matches!(&event.data, Some(Data::Cleared(cleared)) if cleared.ownership.is_none())
        });
        BatchFacts {
            sequence,
            clears,
            ..BatchFacts::default()
        }
    }

    fn batch(data: impl IntoIterator<Item = Data>) -> KvEventBatch {
        KvEventBatch {
            events: data
                .into_iter()
                .map(|data| KvCacheEvent {
                    data: Some(data),
                    ..KvCacheEvent::default()
                })
                .collect(),
            ..KvEventBatch::default()
        }
    }

    fn block(block_hash: i64, tokens: &[u32]) -> KvBlock {
        KvBlock {
            block_hash,
            token_ids: tokens.to_vec(),
            ..KvBlock::default()
        }
    }

    fn stored(hash: i64) -> KvBlocksStored {
        KvBlocksStored {
            blocks: vec![block(hash, &[1, 2])],
            ..KvBlocksStored::default()
        }
    }

    fn removed(hashes: &[i64]) -> KvBlocksRemoved {
        KvBlocksRemoved {
            block_hashes: hashes.to_vec(),
            ..KvBlocksRemoved::default()
        }
    }

    /// The tokens of the block `hash` in a case: its own, so two root blocks
    /// are two blocks of the index.
    fn tokens_of(hash: i64) -> [u32; 2] {
        [7, u32::try_from(hash).unwrap_or_default()]
    }

    /// A main attention store of one block under a group number, as an
    /// engine that names its groups writes it.
    fn main_store(hash: i64, group: u32) -> KvBlocksStored {
        KvBlocksStored {
            blocks: vec![block(hash, &tokens_of(hash))],
            group_idx: Some(group),
            kv_cache_spec_kind: Some("full_attention".into()),
            ..KvBlocksStored::default()
        }
    }

    /// A window group's store of the blocks: with their tokens, or their
    /// hashes alone.
    fn window_store(hashes: &[i64], group: u32, tokens: bool) -> KvBlocksStored {
        KvBlocksStored {
            blocks: hashes
                .iter()
                .map(|&hash| {
                    let own = tokens_of(hash);
                    block(hash, if tokens { &own } else { &[] })
                })
                .collect(),
            group_idx: Some(group),
            kv_cache_spec_kind: Some("sliding_window".into()),
            ..KvBlocksStored::default()
        }
    }

    /// A removal that names its group and no kind, as every removal does.
    fn group_removal(hashes: &[i64], group: u32) -> KvBlocksRemoved {
        KvBlocksRemoved {
            group_idx: Some(group),
            ..removed(hashes)
        }
    }

    fn assert_store_is_skipped(event: KvBlocksStored) {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 0);
    }

    /// A main store that names a namespace, an adapter and a tenant's salt,
    /// is indexed with its chain as any other, over two batches. The hashes
    /// under the namespace's seed are the apply rules' own.
    #[test]
    fn a_main_store_of_a_namespace_is_indexed_with_its_chain() {
        let f = Fixture::new();
        let parent = block(-1, &[1, 2]);
        let parent_hash = parent.block_hash;
        let mut event = KvBlocksStored {
            blocks: vec![parent],
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full_attention".into()),
            lora_name: Some("adapter".into()),
            cache_salt: Some("tenant".into()),
            ..KvBlocksStored::default()
        };
        f.applied(1, &batch([Data::Stored(event.clone())]));
        event.parent_block_hash = Some(parent_hash);
        event.blocks = vec![block(2, &[3, 4]), block(3, &[5, 6])];
        f.applied(2, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 3);
    }

    #[test]
    fn a_foreign_owners_store_is_passed_over() {
        assert_store_is_skipped(KvBlocksStored {
            ownership: Some("KvCr".into()),
            ..stored(1)
        });
    }

    #[test]
    fn a_remote_store_is_passed_over() {
        assert_store_is_skipped(KvBlocksStored {
            locality: Some(KvCacheLocality::Remote as i32),
            ..stored(1)
        });
    }

    #[test]
    fn a_store_on_an_unindexed_tier_is_passed_over() {
        for (tier, level) in [
            (Some(KvCacheTier::Disk as i32), Some(0)),
            (Some(KvCacheTier::External as i32), None),
            (None, Some(2)),
            (Some(KvCacheTier::Unspecified as i32), Some(3)),
            (Some(99), Some(2)),
        ] {
            let mut event = stored(1);
            event.tier = tier;
            event.blocks[0].cache_level = level;
            assert_store_is_skipped(event);
        }
    }

    #[test]
    fn a_store_uses_the_named_tier_then_the_first_blocks_cache_level() {
        for (tier, level) in [
            (Some(KvCacheTier::Device as i32), Some(2)),
            (Some(KvCacheTier::Host as i32), Some(2)),
            (None, Some(1)),
            (Some(KvCacheTier::Unspecified as i32), Some(1)),
            (Some(99), Some(1)),
        ] {
            let f = Fixture::new();
            let mut event = stored(1);
            event.tier = tier;
            event.blocks[0].cache_level = level;
            event.blocks.push(KvBlock {
                cache_level: Some(2),
                ..block(2, &[3, 4])
            });
            f.applied(1, &batch([Data::Stored(event)]));
            assert_eq!(f.blocks(), 2, "tier {tier:?}, cache level {level:?}");
        }
    }

    #[test]
    fn each_main_attention_kind_is_indexed_with_a_group() {
        for kind in ["full_attention", "mla_attention", "sink_full_attention"] {
            let f = Fixture::new();
            let event = KvBlocksStored {
                group_idx: Some(1),
                kv_cache_spec_kind: Some(kind.into()),
                ..stored(1)
            };
            f.applied(1, &batch([Data::Stored(event)]));
            assert_eq!(f.blocks(), 1, "{kind}");
        }
    }

    #[test]
    fn a_kindless_store_of_a_group_not_shown_as_another_kind_is_the_main_groups() {
        let f = Fixture::new();
        let event = KvBlocksStored {
            group_idx: Some(0),
            ..stored(1)
        };
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_kindless_store_of_a_group_shown_as_another_kind_is_passed_over() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(main_store(1, 0)),
                Data::Stored(window_store(&[1], 1, false)),
            ]),
        );
        let event = KvBlocksStored {
            group_idx: Some(1),
            ..stored(2)
        };
        f.applied(2, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_window_group_with_tokens_is_the_main_signal_until_a_main_group_shows() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(window_store(&[1, 2], 1, true))]));
        assert_eq!(f.blocks(), 2);
        // Its kind-less removals follow it to the index.
        f.applied(2, &batch([Data::Removed(group_removal(&[2], 1))]));
        assert_eq!(f.blocks(), 1);
        // A main group shows: the signal's blocks go, the main group's stand.
        f.applied(3, &batch([Data::Stored(main_store(7, 0))]));
        assert_eq!(f.blocks(), 1);
        // The window group is a group of another kind from here.
        f.applied(
            4,
            &batch([
                Data::Stored(window_store(&[8], 1, true)),
                Data::Removed(group_removal(&[7], 1)),
            ]),
        );
        assert_eq!(f.blocks(), 1);
    }

    /// A removal that names a group without its kind is the group's where
    /// the group was shown, even ahead of the group's first store in the
    /// batch, and even for a group no view could place: the main block of
    /// the same hash stays indexed. A main store without tokens indexes
    /// nothing.
    #[test]
    fn a_kind_less_group_event_never_reaches_the_main_index() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(stored(1)),
                Data::Removed(group_removal(&[1], 1)),
                Data::Stored(window_store(&[1], 1, false)),
            ]),
        );
        assert_eq!(f.blocks(), 1, "the main block stays");

        let mut hidden = window_store(&[2], 17, false);
        hidden.kv_cache_spec_kind = Some("mamba".into());
        let second = KvBlocksStored {
            parent_block_hash: Some(1),
            ..stored(2)
        };
        f.applied(2, &batch([Data::Stored(hidden), Data::Stored(second)]));
        assert_eq!(f.blocks(), 2);
        f.applied(3, &batch([Data::Removed(group_removal(&[2], 17))]));
        assert_eq!(f.blocks(), 2, "the main block stays");

        let bare = KvBlocksStored {
            blocks: vec![block(3, &[])],
            ..KvBlocksStored::default()
        };
        f.applied(4, &batch([Data::Stored(bare)]));
        assert_eq!(f.blocks(), 2);
    }

    /// A worker whose attention is all sliding window has no main group:
    /// its window stores carry tokens and are the prefix index's, as a plain
    /// worker's, and a kind-less removal of the group leaves the index.
    #[test]
    fn a_window_only_workers_stores_and_removals_are_the_prefix_indexs() {
        let f = Fixture::new();
        let second = KvBlocksStored {
            parent_block_hash: Some(1),
            ..window_store(&[2], 0, true)
        };
        f.applied(
            1,
            &batch([
                Data::Stored(window_store(&[1], 0, true)),
                Data::Stored(second),
            ]),
        );
        assert_eq!(f.blocks(), 2);
        f.applied(2, &batch([Data::Removed(group_removal(&[2], 0))]));
        assert_eq!(f.blocks(), 1);
    }

    /// A store the index skips (a residency agent's, a remote tier's) shows
    /// no group in a batch: a foreign main store ahead of a window-only
    /// worker's store leaves that store in the prefix index, and a foreign
    /// window store shows no group.
    #[test]
    fn a_foreign_store_in_a_batch_shows_no_group() {
        let f = Fixture::new();
        let agent_main = KvBlocksStored {
            ownership: Some("kvcr".into()),
            ..main_store(7, 1)
        };
        let remote_window = KvBlocksStored {
            locality: Some(KvCacheLocality::Remote as i32),
            ..window_store(&[8], 2, true)
        };
        f.applied(
            1,
            &batch([
                Data::Stored(agent_main),
                Data::Stored(window_store(&[1], 0, true)),
                Data::Stored(remote_window),
            ]),
        );
        assert_eq!(f.blocks(), 1, "the window store is the prefix index's");

        // The group pass's own case: a worker whose main group has shown
        // gets a foreign window store, which would show that group. Skipped,
        // the group stays unshown, and its kind-less removal is the main
        // group's.
        let f = Fixture::new();
        let remote_window = KvBlocksStored {
            locality: Some(KvCacheLocality::Remote as i32),
            ..window_store(&[9], 2, true)
        };
        f.applied(
            1,
            &batch([Data::Stored(stored(1)), Data::Stored(remote_window)]),
        );
        assert_eq!(f.blocks(), 1);
        f.applied(2, &batch([Data::Removed(group_removal(&[1], 2))]));
        assert_eq!(f.blocks(), 0);
    }

    /// A relay snapshot replays the stores after a clear in one batch: the
    /// batch reads as the stream did, and a removal after it still leaves
    /// the index.
    #[test]
    fn a_window_only_workers_snapshot_replay_reads_as_its_stream() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(window_store(&[1], 0, true))]));
        let second = KvBlocksStored {
            parent_block_hash: Some(1),
            ..window_store(&[2], 0, true)
        };
        f.applied(
            2,
            &batch([
                Data::Cleared(KvCacheCleared::default()),
                Data::Stored(window_store(&[1], 0, true)),
                Data::Stored(second),
            ]),
        );
        assert_eq!(f.blocks(), 2);
        f.applied(
            3,
            &batch([
                Data::Removed(group_removal(&[1], 0)),
                Data::Removed(group_removal(&[2], 0)),
            ]),
        );
        assert_eq!(f.blocks(), 0);
    }

    /// A worker whose first batches carried only a window group's stores
    /// (its main signal) and then shows a main group: the index lets the
    /// signal's blocks go and the main group's stores rebuild it, and the
    /// window group's removals are never the main group's blocks' from then
    /// on.
    #[test]
    fn a_main_group_shown_later_moves_the_window_group_to_the_view() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(window_store(&[1], 0, true))]));
        assert_eq!(f.blocks(), 1);

        // The main lane's batch of the same block: the signal's copy goes,
        // the main group's comes.
        f.applied(2, &batch([Data::Stored(main_store(1, 1))]));
        assert_eq!(f.blocks(), 1);

        // The window group's later store and its removals leave the index
        // alone: the main group's blocks stand until the main group lets
        // them go.
        f.applied(3, &batch([Data::Stored(window_store(&[1], 0, false))]));
        f.applied(4, &batch([Data::Stored(main_store(4, 1))]));
        f.applied(
            5,
            &batch([
                Data::Removed(group_removal(&[1], 0)),
                Data::Removed(group_removal(&[4], 0)),
            ]),
        );
        assert_eq!(
            f.blocks(),
            2,
            "the window group's removals leave the index alone"
        );
        f.applied(6, &batch([Data::Removed(group_removal(&[1], 1))]));
        assert_eq!(f.blocks(), 1, "the main group let it go");
    }

    /// A hybrid worker whose window stores carry tokens: the main group's
    /// store in the same batch keeps them out of the prefix index.
    #[test]
    fn a_hybrid_workers_tokenful_window_store_stays_in_the_view() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(window_store(&[1], 1, true)),
                Data::Stored(main_store(1, 0)),
            ]),
        );
        assert_eq!(f.blocks(), 1);
        // The window group's kind-less removal is the group's.
        f.applied(2, &batch([Data::Removed(group_removal(&[1], 1))]));
        assert_eq!(f.blocks(), 1);
    }

    /// A store that names a kind other than main attention and no group is
    /// not applied, and a plain store after it is indexed.
    #[test]
    fn a_nameless_group_store_is_counted_and_a_plain_worker_has_no_view() {
        let f = Fixture::new();
        let nameless = KvBlocksStored {
            kv_cache_spec_kind: Some("sliding_window".into()),
            ..stored(1)
        };
        f.applied(1, &batch([Data::Stored(nameless)]));
        assert_eq!(f.blocks(), 0);
        f.applied(2, &batch([Data::Stored(stored(1))]));
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_batchs_stores_of_other_groups_are_decoded_last() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(main_store(1, 0)),
                Data::Stored(window_store(&[1], 1, false)),
            ]),
        );
        // Group 1 is a group of another kind, so its kind-less store waits
        // for the batch's other events. The main store behind it takes the
        // group back, and the store is the main group's when it is decoded.
        let kindless = KvBlocksStored {
            group_idx: Some(1),
            ..stored(2)
        };
        f.applied(
            2,
            &batch([Data::Stored(kindless), Data::Stored(main_store(3, 1))]),
        );
        assert_eq!(f.blocks(), 3);
    }

    #[test]
    fn a_late_batch_that_shows_a_main_group_is_no_restarted_publisher() {
        let f = Fixture::new();
        f.applied(4, &batch([Data::Stored(window_store(&[1], 0, true))]));
        f.applied(5, &batch([Data::Stored(window_store(&[2], 0, true))]));
        // A main store at a sequence the row holds would restart the worker.
        // Its clear is not the engine's own: the store skips the batch.
        let late = batch([Data::Stored(main_store(7, 1))]);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(4), &late),
            BatchOutcome::Skipped
        );
        assert_eq!(f.blocks(), 2);
        // The window group is the worker's main signal still.
        f.applied(6, &batch([Data::Stored(window_store(&[3], 0, true))]));
        assert_eq!(f.blocks(), 3);
    }

    /// A clear owned by a name the header does not count, at a sequence the
    /// row holds: the batch is late, and the store skips it as any other.
    #[test]
    fn a_late_batch_with_an_owned_clear_is_skipped_as_any_late_batch() {
        let f = Fixture::new();
        f.applied(9, &batch([Data::Stored(stored(1))]));
        let second = KvBlocksStored {
            parent_block_hash: Some(1),
            ..stored(2)
        };
        f.applied(10, &batch([Data::Stored(second)]));
        let late = batch([
            Data::Cleared(KvCacheCleared {
                ownership: Some("engine".into()),
            }),
            Data::Stored(main_store(3, 0)),
        ]);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(9), &late),
            BatchOutcome::Skipped
        );
        assert_eq!(f.blocks(), 2);
    }

    /// The first batch of a new connection under the cursor is a restarted
    /// publisher by the store's own rule. Its owned clear still wipes what
    /// stood ahead of it in the batch.
    #[test]
    fn an_owned_clear_in_a_late_batch_the_store_takes_wipes_what_stood_ahead_of_it() {
        let f = Fixture::new();
        f.applied(9, &batch([Data::Stored(stored(1))]));
        f.sink.reconnected(f.row);
        let restarted = batch([
            Data::Stored(main_store(5, 0)),
            Data::Cleared(KvCacheCleared {
                ownership: Some("engine".into()),
            }),
            Data::Stored(main_store(6, 0)),
        ]);
        f.applied(3, &restarted);
        assert_eq!(f.blocks(), 1);
        // The block that stands is the one behind the clear.
        f.applied(4, &batch([Data::Removed(group_removal(&[6], 0))]));
        assert_eq!(f.blocks(), 0);
    }

    /// Engines number batches from 0, and the row's cursor reads 0 behind
    /// batch 0. A batch 0 that arrives again is late there as at any other
    /// sequence: its owned clear empties nothing.
    #[test]
    fn a_repeated_first_batch_with_an_owned_clear_is_skipped() {
        let f = Fixture::new();
        f.applied(0, &batch([Data::Stored(stored(1))]));
        let late = batch([
            Data::Cleared(KvCacheCleared {
                ownership: Some("engine".into()),
            }),
            Data::Stored(main_store(3, 0)),
        ]);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts_of(0, &late), &late),
            BatchOutcome::Skipped
        );
        // The first batch's block stands: its removal empties the rank.
        assert_eq!(f.blocks(), 1);
        f.applied(1, &batch([Data::Removed(removed(&[1]))]));
        assert_eq!(f.blocks(), 0);
    }

    /// A main store in a batch 0 that arrives again would restart the
    /// worker by the adapter's own clear. Behind batch 0 it is late as
    /// well, and the store skips it.
    #[test]
    fn a_repeated_first_batch_that_shows_a_main_group_is_no_restarted_publisher() {
        let f = Fixture::new();
        f.applied(0, &batch([Data::Stored(window_store(&[1], 0, true))]));
        let late = batch([Data::Stored(main_store(7, 1))]);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts_of(0, &late), &late),
            BatchOutcome::Skipped
        );
        assert_eq!(f.blocks(), 1);
        // The window group is the worker's main signal still.
        f.applied(1, &batch([Data::Stored(window_store(&[2], 0, true))]));
        assert_eq!(f.blocks(), 2);
    }

    /// A cursor reads 0 behind a resync too, and the resync emptied the
    /// rank. A batch 0 there is the stream's first: the store takes it, and
    /// what it holds behind its last clear stands.
    #[test]
    fn a_first_batch_behind_a_resync_stands_from_its_last_clear() {
        let f = Fixture::new();
        f.applied(4, &batch([Data::Stored(stored(1))]));
        f.sink.resync(f.row, None, ResyncReason::DataLoss);
        assert_eq!(f.blocks(), 0);
        let first = batch([
            Data::Stored(main_store(5, 0)),
            Data::Cleared(KvCacheCleared {
                ownership: Some("engine".into()),
            }),
            Data::Stored(main_store(6, 0)),
        ]);
        f.applied(0, &first);
        assert_eq!(f.blocks(), 1);
        f.applied(1, &batch([Data::Removed(group_removal(&[6], 0))]));
        assert_eq!(f.blocks(), 0);
    }

    /// A main group shown in a batch 0 behind a resync: the adapter's own
    /// clear is read as late there, and the resync emptied the rank already.
    #[test]
    fn a_first_batch_behind_a_resync_that_shows_a_main_group_stands() {
        let f = Fixture::new();
        f.applied(4, &batch([Data::Stored(window_store(&[1], 0, true))]));
        f.sink.resync(f.row, None, ResyncReason::DataLoss);
        f.applied(0, &batch([Data::Stored(main_store(7, 1))]));
        assert_eq!(f.blocks(), 1);
        // The window group is a group of another kind now: its store is not
        // the index's.
        f.applied(1, &batch([Data::Stored(window_store(&[2], 0, true))]));
        assert_eq!(f.blocks(), 1);
        // The block that stands is the main group's.
        f.applied(2, &batch([Data::Removed(group_removal(&[7], 1))]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_batch_the_store_does_not_take_shows_nothing() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(main_store(1, 0))]));
        // A batch at a sequence the store holds: its store shows no group.
        let late = batch([Data::Stored(window_store(&[1], 0, false))]);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(1), &late),
            BatchOutcome::Skipped
        );
        f.applied(2, &batch([Data::Removed(group_removal(&[1], 0))]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_worker_that_left_takes_what_its_stream_showed_with_it() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(main_store(1, 0)),
                Data::Stored(window_store(&[1], 1, false)),
            ]),
        );
        f.sink.removed(f.row);
        // Nothing is shown of group 1 now: its kind-less store is the main
        // group's.
        let event = KvBlocksStored {
            group_idx: Some(1),
            ..stored(2)
        };
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_store_with_a_block_without_tokens_is_passed_over() {
        let mut event = stored(1);
        event.blocks.push(block(2, &[]));
        assert_store_is_skipped(event);
    }

    #[test]
    fn a_store_without_blocks_is_passed_over() {
        let f = Fixture::new();
        let empty = batch([Data::Stored(KvBlocksStored::default())]);
        f.applied(1, &empty);
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_removal_takes_the_blocks_out() {
        let f = Fixture::new();
        let mut event = stored(-1);
        event.blocks.push(block(2, &[3, 4]));
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 2);
        // The rules count a block's copies a tier. A removal on the tier
        // that holds no copy of the blocks takes none out.
        let from_host = KvBlocksRemoved {
            tier: Some(KvCacheTier::Host as i32),
            cache_level: Some(2),
            ..removed(&[-1, 2])
        };
        f.applied(2, &batch([Data::Removed(from_host)]));
        assert_eq!(f.blocks(), 2);
        f.applied(3, &batch([Data::Removed(removed(&[-1, 2]))]));
        assert_eq!(f.blocks(), 0);
    }

    /// A block stored on the device and on the host is one block of the
    /// index with two copies. It leaves with its last copy.
    #[test]
    fn a_block_leaves_the_index_with_its_last_copy() {
        let f = Fixture::new();
        let on_host = KvBlocksStored {
            tier: Some(KvCacheTier::Host as i32),
            ..stored(1)
        };
        f.applied(1, &batch([Data::Stored(stored(1)), Data::Stored(on_host)]));
        assert_eq!(f.blocks(), 1);
        f.applied(2, &batch([Data::Removed(removed(&[1]))]));
        assert_eq!(f.blocks(), 1, "the host still holds a copy");
        let from_host = KvBlocksRemoved {
            tier: Some(KvCacheTier::Host as i32),
            ..removed(&[1])
        };
        f.applied(3, &batch([Data::Removed(from_host)]));
        assert_eq!(f.blocks(), 0);
    }

    /// Two copies on one tier need two removals.
    #[test]
    fn two_copies_on_a_tier_need_two_removals() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([Data::Stored(stored(1)), Data::Stored(stored(1))]),
        );
        assert_eq!(f.blocks(), 1);
        f.applied(2, &batch([Data::Removed(removed(&[1]))]));
        assert_eq!(f.blocks(), 1, "the other copy is still cached");
        f.applied(3, &batch([Data::Removed(removed(&[1]))]));
        assert_eq!(f.blocks(), 0);
    }

    /// The store empties a rank in its own turn, here at a resync. The
    /// row's next batch reads another value of its lane and forgets the
    /// copies it counted: a block stored once behind the resync leaves at
    /// its one removal.
    #[test]
    fn a_rank_the_store_emptied_forgets_the_copies_it_counted() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([Data::Stored(stored(1)), Data::Stored(stored(1))]),
        );
        f.store
            .resync(f.row, Some(RANK), store::ResyncReason::DataLoss);
        assert_eq!(f.blocks(), 0);
        f.applied(5, &batch([Data::Stored(stored(1))]));
        assert_eq!(f.blocks(), 1);
        f.applied(6, &batch([Data::Removed(removed(&[1]))]));
        assert_eq!(f.blocks(), 0, "one copy stood behind the resync");
    }

    /// A store narrower than the row's block of record is left out: an
    /// engine's prefix entry at its hash granularity, which no request
    /// chunked at the block matches.
    #[test]
    fn a_store_narrower_than_the_block_of_record_is_left_out() {
        let f = Fixture::new();
        f.applied(1, &wide(1, 32));
        assert_eq!(f.blocks(), 1);
        f.applied(2, &wide(77, 4));
        assert_eq!(f.blocks(), 1);
        // A wider store raises the record, and what was indexed stays.
        f.applied(3, &wide(2, 64));
        assert_eq!(f.blocks(), 2);
        f.applied(4, &wide(3, 32));
        assert_eq!(f.blocks(), 2);
    }

    /// The store empties a rank in its own turn, and the row's next batch
    /// forgets its block of record: an engine that comes back with a
    /// narrower block has its stores indexed as they come.
    #[test]
    fn a_rank_the_store_emptied_forgets_its_block_of_record() {
        let f = Fixture::new();
        f.applied(1, &wide(1, 32));
        f.store
            .resync(f.row, Some(RANK), store::ResyncReason::DataLoss);
        f.applied(5, &wide(2, 16));
        assert_eq!(f.blocks(), 1, "the narrower block is the row's block now");
    }

    /// The row's view in the store's index, read by a lent batch with no
    /// event at `sequence`.
    fn view(f: &Fixture, sequence: u64) -> Arc<WorkerGroups> {
        let facts = store::BatchFacts {
            sequence,
            ..store::BatchFacts::default()
        };
        let mut view = None;
        let outcome = f
            .store
            .admit_through(f.row, RANK, facts, &mut |lane| view = Some(lane.groups()));
        assert_eq!(outcome, store::BatchOutcome::Applied);
        view.expect("the body ran")
    }

    /// A main store and a window group's store of the same block, as a
    /// hybrid engine writes them.
    fn hybrid() -> KvEventBatch {
        batch([
            Data::Stored(main_store(1, 0)),
            Data::Stored(window_store(&[1], 1, false)),
        ])
    }

    /// A store of a cache group other than main attention feeds the row's
    /// view in the store's index, where a decision can read it.
    #[test]
    fn a_group_store_feeds_the_rows_view_in_the_stores_index() {
        let f = Fixture::new();
        f.applied(1, &hybrid());
        assert_eq!(f.blocks(), 1);
        let view = view(&f, 2);
        assert!(view.knows(1), "the window group is the view's");
        assert_eq!(view.named_blocks(), 1, "the main block is named in it");
    }

    /// The store makes a row's lane again under the row's id, and its index
    /// holds another view for the row. The row's next batch reads another
    /// value of its lane and takes that view: what its group stores name
    /// is in the store's index, not in the view of the lane before.
    #[test]
    fn a_lane_made_again_gives_the_rows_state_the_indexs_new_view() {
        let f = Fixture::new();
        f.applied(1, &hybrid());
        let old = view(&f, 2);
        // The row stops asking for its events, and asks again.
        let facts = f.store.get(f.row).expect("the row").facts;
        let quiet = WorkerFacts {
            signals: Signals::default(),
            ..facts.clone()
        };
        assert_eq!(f.store.upsert(quiet).id(), f.row);
        assert_eq!(f.store.upsert(facts).id(), f.row);
        f.applied(1, &hybrid());
        let new = view(&f, 2);
        assert!(!Arc::ptr_eq(&old, &new));
        assert!(new.knows(1), "the new view is the one the row writes");
        assert_eq!(new.named_blocks(), 1);
    }

    #[test]
    fn a_removal_of_the_main_group_is_taken_under_its_group_number() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(main_store(1, 0))]));
        f.applied(2, &batch([Data::Removed(group_removal(&[1], 0))]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_foreign_remote_or_unindexed_removal_is_passed_over() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        let options = [
            (Some("KvCr"), None, None, None),
            (None, Some(KvCacheLocality::Remote as i32), None, None),
            (None, None, Some(KvCacheTier::Disk as i32), Some(0)),
            (None, None, None, Some(2)),
        ];
        for (sequence, (owner, locality, tier, cache_level)) in (2..).zip(options) {
            let removal = KvBlocksRemoved {
                ownership: owner.map(str::to_owned),
                locality,
                tier,
                cache_level,
                ..removed(&[1])
            };
            f.applied(sequence, &batch([Data::Removed(removal)]));
            assert_eq!(f.blocks(), 1, "removal at sequence {sequence}");
        }
    }

    #[test]
    fn a_clear_empties_the_rank() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        assert_eq!(f.blocks(), 1);
        f.applied(2, &batch([Data::Cleared(KvCacheCleared::default())]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_foreign_owners_clear_is_passed_over() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        f.applied(
            2,
            &batch([Data::Cleared(KvCacheCleared {
                ownership: Some("KvCr".into()),
            })]),
        );
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn an_event_with_no_data_gives_none() {
        let f = Fixture::new();
        let missing = KvEventBatch {
            events: vec![KvCacheEvent::default()],
            ..KvEventBatch::default()
        };
        f.applied(1, &missing);
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn the_order_of_a_batch_of_three_event_kinds_is_kept() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        let ordered = batch([
            Data::Removed(removed(&[1])),
            Data::Cleared(KvCacheCleared::default()),
            Data::Stored(stored(2)),
        ]);
        // The store behind the clear stands: the three were applied in the
        // batch's order.
        f.applied(2, &ordered);
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn the_facts_and_the_outcome_cross_as_they_are() {
        let f = Fixture::new();
        f.applied(21, &batch([Data::Stored(stored(1))]));
        let empty = batch([]);
        f.applied(22, &empty);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(26), &empty),
            BatchOutcome::Gap {
                expected: 23,
                received: 26
            }
        );
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(22), &empty),
            BatchOutcome::Skipped
        );

        let clear = BatchFacts {
            clears: true,
            ..facts(20)
        };
        assert_eq!(
            f.sink.admit(f.row, RANK, clear, &empty),
            BatchOutcome::Applied
        );
        assert_eq!(f.blocks(), 0);
        for (sequence, unknown_before) in [(70, true), (71, false)] {
            let chunk = BatchFacts {
                snapshot: Some(SnapshotChunk { unknown_before }),
                ..facts(sequence)
            };
            assert_eq!(
                f.sink.admit(f.row, RANK, chunk, &empty),
                BatchOutcome::Applied
            );
            assert_eq!(
                f.store.cursor(f.row, RANK),
                Some(store::KvCursor {
                    applied: sequence,
                    degraded: unknown_before,
                    replay_pending: false
                })
            );
        }
    }

    #[test]
    fn each_reason_reaches_the_store_under_its_own_name() {
        use store::ResyncReason as To;

        let f = Fixture::new();
        let store = batch([Data::Stored(stored(1))]);
        for (reason, expected) in [
            (ResyncReason::OutOfRange, To::OutOfRange),
            (ResyncReason::DataLoss, To::DataLoss),
            (ResyncReason::SnapshotAbandoned, To::SnapshotAbandoned),
            (ResyncReason::PublisherRestart, To::PublisherRestart),
            (ResyncReason::GapCleared, To::GapCleared),
            (ResyncReason::Snapshot, To::Snapshot),
        ] {
            f.sink.resync(f.row, Some(RANK), reason);
            assert_eq!(f.store.resyncs(expected), 1, "{reason:?}");

            f.applied(7, &store);
            f.sink.resync(f.row, Some(0), reason);
            assert_eq!(f.blocks(), 1);
            f.sink.resync(f.row, Some(RANK), reason);
            // An abandoned snapshot keeps the block it showed until the
            // next reason empties the rank. Every other reason empties it.
            let kept = usize::from(reason == ResyncReason::SnapshotAbandoned);
            assert_eq!(f.blocks(), kept, "{reason:?}");
        }

        f.applied(7, &store);
        f.sink.resync(f.row, None, ResyncReason::OutOfRange);
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_cursor_reads_as_the_stores() {
        let f = Fixture::new();
        assert_eq!(f.sink.cursor(f.row, RANK), None);
        f.store.admit(
            f.row,
            RANK,
            store::BatchFacts {
                sequence: 41,
                clears: false,
                snapshot: Some(store::SnapshotChunk {
                    unknown_before: true,
                }),
            },
            &[],
        );
        f.store.admit(
            f.row,
            RANK,
            store::BatchFacts {
                sequence: 43,
                ..Default::default()
            },
            &[],
        );
        let expected = f.store.cursor(f.row, RANK).expect("the store's cursor");
        assert!(expected.replay_pending && expected.degraded);
        assert_eq!(
            f.sink.cursor(f.row, RANK),
            Some(KvCursor {
                applied: expected.applied,
                replay_pending: expected.replay_pending,
                degraded: expected.degraded,
            })
        );
        assert_eq!(f.sink.cursor(f.row, 0), None);
    }

    #[test]
    fn a_reconnection_is_reported_to_the_store() {
        let f = Fixture::new();
        f.applied(7, &batch([Data::Stored(stored(1))]));
        f.sink.reconnected(f.row);
        f.applied(6, &batch([]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_worker_reported_gone_loses_its_blocks_and_cursor() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        assert_eq!(f.blocks(), 1);
        f.sink.removed(f.row);
        assert_eq!(f.blocks(), 0);
        assert_eq!(f.store.cursor(f.row, RANK), None);
    }

    /// The feed of the fixture's one worker, opened by its host.
    fn opened(f: &Fixture) -> (KvHost<Members, Store>, Arc<KvAdapter<Store>>) {
        let host = f.host();
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the worker's object");
        let ticket = host.enroll(&worker, MODEL);
        let (sink, row) = host.open(ticket, &worker, MODEL).expect("the row's feed");
        assert_eq!(row, f.row);
        (host, sink)
    }

    /// One store of one block `block_size` tokens wide.
    fn wide_store(hash: i64, block_size: i32) -> KvBlocksStored {
        KvBlocksStored {
            blocks: vec![KvBlock {
                block_size,
                ..block(hash, &tokens_of(hash))
            }],
            ..KvBlocksStored::default()
        }
    }

    /// One store of one block `block_size` tokens wide, alone in its batch.
    fn wide(hash: i64, block_size: i32) -> KvEventBatch {
        batch([Data::Stored(wide_store(hash, block_size))])
    }

    /// A batch the store takes names the model's block size, whatever the
    /// worker stated at its registration: none, a narrower size or a wider
    /// one. A narrower store behind it moves nothing, and a batch the store
    /// skips names none.
    #[test]
    fn a_batch_the_store_takes_names_the_models_block_size() {
        for stated in [None, Some(16), Some(64)] {
            let f = Fixture::single(stated);
            let (_host, sink) = opened(&f);
            assert_eq!(f.store.block_size(MODEL), stated);
            let mut first = wide(1, 16);
            first.events.extend(wide(2, 32).events);
            assert_eq!(
                sink.admit(f.row, 0, facts(1), &first),
                BatchOutcome::Applied
            );
            assert_eq!(f.store.block_size(MODEL), Some(32), "stated: {stated:?}");
            // A narrower store is a prefix entry.
            assert_eq!(
                sink.admit(f.row, 0, facts(2), &wide(3, 8)),
                BatchOutcome::Applied
            );
            assert_eq!(f.store.block_size(MODEL), Some(32));
            // A duplicate of the batch the row holds is skipped.
            assert_eq!(
                sink.admit(f.row, 0, facts(2), &wide(4, 64)),
                BatchOutcome::Skipped
            );
            assert_eq!(f.store.block_size(MODEL), Some(32));
        }
    }

    /// A store the index passes over names no size. One on a tier the index
    /// does not track is wider than the engine's block here, and the size of
    /// record stays the indexed store's.
    #[test]
    fn a_store_the_index_passes_over_names_no_block_size() {
        let f = Fixture::single(None);
        let (_host, sink) = opened(&f);
        let passed_over = KvBlocksStored {
            tier: Some(KvCacheTier::Disk as i32),
            ..wide_store(2, 64)
        };
        let both = batch([Data::Stored(wide_store(1, 16)), Data::Stored(passed_over)]);
        assert_eq!(sink.admit(f.row, 0, facts(1), &both), BatchOutcome::Applied);
        assert_eq!(f.store.block_size(MODEL), Some(16));
    }

    /// A registration that names a wider size behind a stored block is put
    /// right by the next stored block, a prefix entry too: the engine's word
    /// only rises. A model that left learns its size anew.
    #[test]
    fn a_stored_block_keeps_the_size_of_record_the_engines_own() {
        let f = Fixture::single(None);
        let (host, sink) = opened(&f);
        let mut sequence: u32 = 0;
        let mut stored = |block_size: i32| {
            sequence += 1;
            let batch = wide(i64::from(sequence), block_size);
            assert_eq!(
                sink.admit(f.row, 0, facts(u64::from(sequence)), &batch),
                BatchOutcome::Applied
            );
            f.store.block_size(MODEL)
        };
        assert_eq!(stored(16), Some(16));
        // Another worker of the model registers and states a wider size.
        f.store.set_block_size(MODEL, 64);
        assert_eq!(f.store.block_size(MODEL), Some(64));
        assert_eq!(stored(16), Some(16));
        // The model leaves, and comes back with a registration's size.
        host.model_left(MODEL);
        assert_eq!(f.store.block_size(MODEL), None);
        f.store.set_block_size(MODEL, 64);
        assert_eq!(stored(16), Some(16));
        // A wider block raises it.
        assert_eq!(stored(32), Some(32));
        // A prefix entry behind a wider registration puts the engine's block
        // back, not its own width.
        f.store.set_block_size(MODEL, 64);
        assert_eq!(stored(8), Some(32));
    }

    #[test]
    fn the_block_size_is_read_and_set_at_the_store() {
        let f = Fixture::new();
        f.store.set_block_size(MODEL, 16);
        assert_eq!(f.sink.block_size(MODEL), Some(16));
        f.sink.set_block_size(MODEL, 32);
        assert_eq!(f.store.block_size(MODEL), Some(32));
    }

    #[test]
    fn a_worker_with_a_row_is_enrolled_under_the_rows_id_and_seeds_the_block_size() {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let host = f.host();
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the worker's object");
        let ticket = host.enroll(&worker, MODEL);
        assert_eq!(ticket, Ok(f.row));
        assert_eq!(f.store.block_size(MODEL), Some(BLOCK_SIZE));
        let (sink, row) = host.open(ticket, &worker, MODEL).expect("the row's feed");
        assert_eq!(row, f.row);
        assert!(Arc::ptr_eq(&sink, &host.sink));
        assert_eq!(
            host.skipped_subscriptions(),
            SkippedSubscriptions::default()
        );
    }

    #[test]
    fn a_row_the_adapter_never_fed_begins_from_nothing() {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let stores = batch([
            Data::Stored(main_store(1, 0)),
            Data::Stored(window_store(&[1], 2, false)),
        ]);
        assert_eq!(
            f.sink.admit(f.row, 0, facts(7), &stores),
            BatchOutcome::Applied
        );
        assert_eq!(
            f.store.cursor(f.row, 0).map(|cursor| cursor.applied),
            Some(7)
        );
        assert_eq!(f.store.indexed_blocks(f.row, 0), 1);

        let host = f.host();
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the worker's object");
        let (sink, row) = host
            .open(host.enroll(&worker, MODEL), &worker, MODEL)
            .expect("the row's feed");
        assert_eq!(sink.cursor(row, 0), None);
        assert_eq!(f.store.indexed_blocks(row, 0), 0);
        assert!(begun(&sink, row));

        assert_eq!(sink.admit(row, 0, facts(9), &stores), BatchOutcome::Applied);
        let (sink, row) = host
            .open(host.enroll(&worker, MODEL), &worker, MODEL)
            .expect("the row's feed");
        assert_eq!(sink.cursor(row, 0).map(|cursor| cursor.applied), Some(9));
        assert_eq!(f.store.indexed_blocks(row, 0), 1);
        assert_eq!(
            sink.admit(
                row,
                0,
                facts(10),
                &batch([Data::Removed(group_removal(&[1], 2))])
            ),
            BatchOutcome::Applied
        );
        assert_eq!(f.store.indexed_blocks(row, 0), 1);
    }

    /// The store keeps a row's lane at the row's own rank. A row the holder
    /// stands above rank 0 is its engine's rank 0 stream's. A ticket for
    /// it opens no feed, and what the store keeps of the row stands. It is
    /// counted as an upper rank's: the holder keeps the row.
    #[test]
    fn a_ticket_for_a_row_above_rank_0_opens_no_feed() {
        let f = Fixture::new();
        let stores = batch([
            Data::Stored(main_store(1, 0)),
            Data::Stored(window_store(&[1], 2, false)),
        ]);
        f.applied(7, &stores);
        assert_eq!(
            f.store.cursor(f.row, RANK).map(|cursor| cursor.applied),
            Some(7)
        );
        assert_eq!(f.blocks(), 1);

        // A host over the same store: its holder stands the row at rank 1,
        // and the row's object stands for it.
        let host = f.host();
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the worker's object");
        assert!(host.open(Ok(f.row), &worker, MODEL).is_none());
        assert_eq!(
            f.store.cursor(f.row, RANK).map(|cursor| cursor.applied),
            Some(7)
        );
        assert_eq!(f.blocks(), 1);
        let unfed = host.skipped_subscriptions();
        assert_eq!((unfed.no_row, unfed.upper_rank), (0, 1));
    }

    #[test]
    fn a_row_above_rank_0_opens_no_feed_and_is_counted() {
        let f = Fixture::new();
        let host = f.host();
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the worker's object");
        let ticket = host.enroll(&worker, MODEL);
        assert_eq!(ticket, Err(SkipReason::UpperRank));
        // The model's size is seeded all the same.
        assert_eq!(f.store.block_size(MODEL), Some(BLOCK_SIZE));
        assert!(host.open(ticket, &worker, MODEL).is_none());
        let unfed = SkippedSubscriptions {
            no_row: 0,
            upper_rank: 1,
        };
        assert_eq!(host.skipped_subscriptions(), unfed);
    }

    #[test]
    fn an_object_the_holder_replaced_opens_nothing() {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let host = f.host();
        let registry = host.objects.registry();
        let old = registry.get_by_url(URL).expect("the worker's object");
        // The row's facts move, so the holder lays another object in the
        // place of the first.
        f.store.upsert(WorkerFacts {
            key: URL.into(),
            engine: URL.into(),
            incarnation: 1,
            models: vec![MODEL.into()],
            dp_rank: 0,
            dp_size: 1,
            kv_block_size: Some(BLOCK_SIZE * 2),
            signals: Signals {
                events: true,
                ..Signals::default()
            },
            ..WorkerFacts::default()
        });
        host.objects.follow(f.store.as_ref());
        let new = registry.get_by_url(URL).expect("the row's new object");
        assert!(!Arc::ptr_eq(&old, &new));
        assert_eq!(host.enroll(&old, MODEL), Err(SkipReason::NoRow));
        assert_eq!(host.enroll(&new, MODEL), Ok(f.row));
    }

    #[test]
    fn a_worker_the_holder_did_not_lay_opens_nothing() {
        let f = Fixture::new();
        let host = f.host();
        let worker: Arc<dyn Worker> = Arc::new(<BasicWorkerBuilder>::new("http://b:8000").build());
        assert!(host
            .objects
            .registry()
            .register(Arc::clone(&worker))
            .is_some());
        let ticket = host.enroll(&worker, MODEL);
        assert_eq!(ticket, Err(SkipReason::NoRow));
        assert!(host.open(ticket, &worker, MODEL).is_none());
        let unfed = SkippedSubscriptions {
            no_row: 1,
            upper_rank: 0,
        };
        assert_eq!(host.skipped_subscriptions(), unfed);
    }

    #[test]
    fn a_worker_with_no_positive_block_size_leaves_it_unset() {
        for size in [None, Some(0)] {
            let f = Fixture::single(size);
            let host = f.host();
            let worker = host
                .objects
                .registry()
                .get_by_url(URL)
                .expect("the worker's object");
            assert_eq!(host.enroll(&worker, MODEL), Ok(f.row));
            assert_eq!(f.store.block_size(MODEL), None);
        }
    }

    #[test]
    fn a_model_that_left_forgets_its_block_size() {
        let f = Fixture::new();
        let host = f.host();
        f.store.set_block_size(MODEL, 32);
        host.model_left(MODEL);
        assert_eq!(f.store.block_size(MODEL), None);
        f.store.set_block_size(MODEL, 16);
        assert_eq!(f.store.block_size(MODEL), Some(16));
    }

    const UPPER: &str = "http://a:8000#1";

    const OTHER: &str = "http://b:8000";
    const THIRD: &str = "http://a:8000#2";
    const FOURTH: &str = "http://b:8000#1";

    /// The row of one rank of the engine at `engine`, which asks for events.
    fn rank_facts(key: &str, engine: &str, dp_rank: u16) -> WorkerFacts {
        WorkerFacts {
            key: key.into(),
            engine: engine.into(),
            incarnation: 1,
            models: vec![MODEL.into()],
            dp_rank,
            dp_size: 3,
            signals: Signals {
                events: true,
                ..Signals::default()
            },
            ..WorkerFacts::default()
        }
    }

    /// The cache-event series of the fixture's worker, read from a recorder
    /// of the case's own. The recorder's crates come with the controller.
    #[cfg(feature = "controller")]
    mod series {
        use metrics_exporter_prometheus::{
            PrometheusBuilder, PrometheusHandle, PrometheusRecorder,
        };

        use super::*;

        struct Series {
            recorder: PrometheusRecorder,
            handle: PrometheusHandle,
        }

        impl Series {
            fn new() -> Self {
                let recorder = PrometheusBuilder::new().build_recorder();
                let handle = recorder.handle();
                Self { recorder, handle }
            }

            /// The value of `name` for the fixture's worker under `labels`,
            /// where the series stands.
            fn read(&self, name: &str, labels: &[&str]) -> Option<f64> {
                let worker = format!("worker=\"{URL}\"");
                let open = format!("{name}{{");
                self.handle.render().lines().find_map(|line| {
                    let (series, value) = line.rsplit_once(' ')?;
                    let named = series.starts_with(&open) && series.contains(&worker);
                    let all = labels.iter().all(|label| series.contains(label));
                    (named && all).then(|| value.parse().ok()).flatten()
                })
            }

            /// Whether any cache-event series names the fixture's worker.
            /// The worker's object has series of its own.
            fn names_the_worker(&self) -> bool {
                let worker = format!("worker=\"{URL}\"");
                let rendered = self.handle.render();
                let mut lines = rendered.lines();
                lines.any(|line| line.starts_with("smg_kv_") && line.contains(&worker))
            }
        }

        /// A batch the store takes counts its blocks by operation and
        /// itself as applied, publishes its pace and sets the worker's
        /// index blocks. A removal behind it counts its blocks and moves
        /// the gauge.
        #[test]
        fn an_applied_batch_counts_its_blocks_and_sets_the_workers_gauge() {
            let series = Series::new();
            metrics::with_local_recorder(&series.recorder, || {
                let f = Fixture::single(Some(BLOCK_SIZE));
                let (_host, sink) = opened(&f);
                let mut first = wide(1, 32);
                first.events.extend(wide(2, 32).events);
                assert_eq!(
                    sink.admit(f.row, 0, facts(1), &first),
                    BatchOutcome::Applied
                );
                let stored = series.read("smg_kv_event_blocks_total", &["op=\"stored\""]);
                assert_eq!(stored, Some(2.0));
                let applied = &["disposition=\"applied\""];
                assert_eq!(
                    series.read("smg_kv_event_batches_total", applied),
                    Some(1.0)
                );
                assert_eq!(series.read("smg_kv_index_blocks", &[]), Some(2.0));
                assert_eq!(
                    series.read("smg_kv_event_apply_seconds_count", &[]),
                    Some(1.0)
                );

                let gone = batch([Data::Removed(removed(&[2]))]);
                assert_eq!(sink.admit(f.row, 0, facts(2), &gone), BatchOutcome::Applied);
                let removed = series.read("smg_kv_event_blocks_total", &["op=\"removed\""]);
                assert_eq!(removed, Some(1.0));
                assert_eq!(
                    series.read("smg_kv_event_batches_total", applied),
                    Some(2.0)
                );
                assert_eq!(series.read("smg_kv_index_blocks", &[]), Some(1.0));
            });
        }

        /// A snapshot chunk counts its blocks and itself under the
        /// snapshot's own words, and publishes no pace.
        #[test]
        fn a_snapshot_chunk_counts_under_its_own_words_and_publishes_no_pace() {
            let series = Series::new();
            metrics::with_local_recorder(&series.recorder, || {
                let f = Fixture::single(Some(BLOCK_SIZE));
                let (_host, sink) = opened(&f);
                let chunk = BatchFacts {
                    snapshot: Some(SnapshotChunk {
                        unknown_before: false,
                    }),
                    ..facts(7)
                };
                assert_eq!(
                    sink.admit(f.row, 0, chunk, &wide(1, 32)),
                    BatchOutcome::Applied
                );
                let stored = series.read("smg_kv_event_blocks_total", &["op=\"snapshot\""]);
                assert_eq!(stored, Some(1.0));
                let chunks = &["disposition=\"snapshot\""];
                assert_eq!(series.read("smg_kv_event_batches_total", chunks), Some(1.0));
                assert_eq!(series.read("smg_kv_index_blocks", &[]), Some(1.0));
                assert_eq!(series.read("smg_kv_event_apply_seconds_count", &[]), None);
            });
        }

        /// A batch behind its lane's cursor counts as stale, and a gap
        /// counts the replay the collector asks.
        #[test]
        fn a_stale_batch_and_a_gap_count_under_their_own_words() {
            let series = Series::new();
            metrics::with_local_recorder(&series.recorder, || {
                let f = Fixture::single(Some(BLOCK_SIZE));
                let (_host, sink) = opened(&f);
                assert_eq!(
                    sink.admit(f.row, 0, facts(1), &wide(1, 32)),
                    BatchOutcome::Applied
                );
                assert_eq!(
                    sink.admit(f.row, 0, facts(1), &wide(1, 32)),
                    BatchOutcome::Skipped
                );
                let stale = &["disposition=\"stale\""];
                assert_eq!(series.read("smg_kv_event_batches_total", stale), Some(1.0));
                assert_eq!(
                    sink.admit(f.row, 0, facts(3), &wide(3, 32)),
                    BatchOutcome::Gap {
                        expected: 2,
                        received: 3
                    }
                );
                let asked = &["outcome=\"replay_requested\""];
                assert_eq!(series.read("smg_kv_event_gaps_total", asked), Some(1.0));
                // Neither moved the applied count or the stored blocks.
                let applied = &["disposition=\"applied\""];
                assert_eq!(
                    series.read("smg_kv_event_batches_total", applied),
                    Some(1.0)
                );
                let stored = series.read("smg_kv_event_blocks_total", &["op=\"stored\""]);
                assert_eq!(stored, Some(1.0));
            });
        }

        /// A reset counts under its reason, each reason by itself.
        #[test]
        fn a_reset_counts_under_its_reason() {
            let series = Series::new();
            metrics::with_local_recorder(&series.recorder, || {
                let f = Fixture::single(Some(BLOCK_SIZE));
                let (_host, sink) = opened(&f);
                sink.resync(f.row, None, ResyncReason::DataLoss);
                sink.resync(f.row, Some(0), ResyncReason::Snapshot);
                sink.resync(f.row, None, ResyncReason::DataLoss);
                let lost = &["reason=\"data_loss\""];
                assert_eq!(series.read("smg_kv_event_resyncs_total", lost), Some(2.0));
                let snapshot = &["reason=\"snapshot\""];
                assert_eq!(
                    series.read("smg_kv_event_resyncs_total", snapshot),
                    Some(1.0)
                );
            });
        }

        /// A store whose parent the row does not hold is placed as a chain
        /// of its own and counted for the worker: the rules' own count
        /// reaches a series.
        #[test]
        fn a_parentless_store_is_counted_for_the_worker() {
            let series = Series::new();
            metrics::with_local_recorder(&series.recorder, || {
                let f = Fixture::single(Some(BLOCK_SIZE));
                let (_host, sink) = opened(&f);
                let orphan = KvBlocksStored {
                    parent_block_hash: Some(99),
                    ..wide_store(1, 32)
                };
                let orphan = batch([Data::Stored(orphan)]);
                assert_eq!(
                    sink.admit(f.row, 0, facts(1), &orphan),
                    BatchOutcome::Applied
                );
                let stores = series.read("smg_kv_event_parentless_stores_total", &[]);
                assert_eq!(stores, Some(1.0));
                let blocks = series.read("smg_kv_event_parentless_blocks_total", &[]);
                assert_eq!(blocks, Some(1.0));
            });
        }

        /// A row no feed was opened for names no series: its batch is
        /// applied by the rules alone. A feed that left names none either.
        #[test]
        fn a_row_with_no_open_feed_names_no_series() {
            let series = Series::new();
            metrics::with_local_recorder(&series.recorder, || {
                let f = Fixture::single(Some(BLOCK_SIZE));
                assert_eq!(
                    f.sink.admit(f.row, 0, facts(1), &wide(1, 32)),
                    BatchOutcome::Applied
                );
                assert_eq!(f.store.indexed_blocks(f.row, 0), 1);
                assert!(!series.names_the_worker());

                let (_host, sink) = opened(&f);
                sink.removed(f.row);
                sink.resync(f.row, None, ResyncReason::DataLoss);
                assert!(!series.names_the_worker());
            });
        }
    }

    /// An engine of three ranks: its rank 0 row, the host that follows the
    /// store, and the feed the host opened for that row.
    struct Ranked {
        store: Arc<MemoryStore>,
        host: KvHost<Members, Store>,
        sink: Arc<KvAdapter<Store>>,
        lower: WorkerId,
    }

    impl Ranked {
        /// Rank 0's row alone: rank 1's is laid by [`Self::upper`].
        fn new() -> Self {
            let store = Arc::new(MemoryStore::new());
            let lower = store.upsert(rank_facts(URL, URL, 0)).id();
            let objects = Arc::new(Objects::new(Arc::new(WorkerRegistry::new())));
            objects.follow(store.as_ref());
            let host = KvHost::new(objects, Arc::clone(&store) as Arc<Store>);
            let worker = host
                .objects
                .registry()
                .get_by_url(URL)
                .expect("the rank 0 row's object");
            let (sink, row) = host
                .open(host.enroll(&worker, MODEL), &worker, MODEL)
                .expect("the rank 0 row's feed");
            assert_eq!(row, lower);
            Self {
                store,
                host,
                sink,
                lower,
            }
        }

        /// Rank 1's row, laid where it is not yet.
        fn upper(&self) -> WorkerId {
            let upper = self.store.upsert(rank_facts(UPPER, URL, 1)).id();
            self.host.objects.follow(self.store.as_ref());
            upper
        }

        /// The rank 0 row of another engine at `key`, laid, with its feed
        /// opened by the host.
        fn other(&self, key: &str) -> WorkerId {
            let row = self.store.upsert(rank_facts(key, key, 0)).id();
            self.host.objects.follow(self.store.as_ref());
            let worker = self
                .host
                .objects
                .registry()
                .get_by_url(key)
                .expect("the other engine's object");
            let (_, opened) = self
                .host
                .open(self.host.enroll(&worker, MODEL), &worker, MODEL)
                .expect("the other engine's feed");
            assert_eq!(opened, row);
            row
        }

        /// One batch of `rank` on the engine's stream, at `sequence`.
        fn admit(&self, rank: u16, sequence: u64, data: Data) -> BatchOutcome {
            self.sink
                .admit(self.lower, rank, facts(sequence), &batch([data]))
        }

        /// The blocks rank 0's row and rank 1's row hold.
        fn blocks(&self, upper: WorkerId) -> (usize, usize) {
            (
                self.store.indexed_blocks(self.lower, 0),
                self.store.indexed_blocks(upper, 1),
            )
        }
    }

    /// A batch of rank 1 on the engine's one stream is indexed under rank
    /// 1's row, and a batch of rank 0 under the stream's own.
    #[test]
    fn a_batch_of_rank_one_is_indexed_under_rank_ones_row() {
        let f = Ranked::new();
        let upper = f.upper();
        let store = |hash| Data::Stored(wide_store(hash, 16));
        assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
        assert_eq!(f.blocks(upper), (0, 1));
        assert_eq!(f.admit(0, 1, store(2)), BatchOutcome::Applied);
        assert_eq!(f.admit(0, 2, store(3)), BatchOutcome::Applied);
        assert_eq!(f.blocks(upper), (2, 1));
    }

    /// A batch of a rank whose row the holder has not laid is passed over,
    /// and the same batch is taken once the row is laid.
    #[test]
    fn a_batch_of_a_rank_with_no_row_is_passed_over() {
        let f = Ranked::new();
        let store = Data::Stored(wide_store(1, 16));
        assert_eq!(f.admit(1, 1, store.clone()), BatchOutcome::Skipped);
        assert_eq!(f.sink.cursor(f.lower, 1), None);
        let upper = f.upper();
        assert_eq!(f.admit(1, 1, store), BatchOutcome::Applied);
        assert_eq!(f.blocks(upper), (0, 1));
    }

    /// A rank's cursor and its reset are its own row's. A rank with no row
    /// has no cursor and resets nothing. The stream's reset reaches every
    /// rank it fed.
    #[test]
    fn a_ranks_cursor_and_resync_reach_its_own_row() {
        let f = Ranked::new();
        let upper = f.upper();
        let store = |hash| Data::Stored(wide_store(hash, 16));
        assert_eq!(f.admit(0, 3, store(1)), BatchOutcome::Applied);
        assert_eq!(f.admit(1, 7, store(2)), BatchOutcome::Applied);
        let applied = |rank| f.sink.cursor(f.lower, rank).map(|cursor| cursor.applied);
        assert_eq!(
            (applied(0), applied(1), applied(2)),
            (Some(3), Some(7), None)
        );

        f.sink.resync(f.lower, Some(1), ResyncReason::GapCleared);
        assert_eq!(f.blocks(upper), (1, 0));
        f.sink.resync(f.lower, Some(2), ResyncReason::GapCleared);
        assert_eq!(f.store.resyncs(store::ResyncReason::GapCleared), 1);

        assert_eq!(f.admit(1, 8, store(3)), BatchOutcome::Applied);
        assert_eq!(f.blocks(upper), (1, 1));
        f.sink.resync(f.lower, None, ResyncReason::OutOfRange);
        assert_eq!(f.blocks(upper), (0, 0));
        assert_eq!(f.store.resyncs(store::ResyncReason::OutOfRange), 2);
    }

    /// An engine that leaves clears every rank its stream fed, and the
    /// adapter keeps nothing of them.
    #[test]
    fn a_removed_engine_clears_every_rank() {
        let f = Ranked::new();
        let upper = f.upper();
        let store = |hash| Data::Stored(wide_store(hash, 16));
        assert_eq!(f.admit(0, 1, store(1)), BatchOutcome::Applied);
        assert_eq!(f.admit(1, 1, store(2)), BatchOutcome::Applied);
        assert_eq!(f.blocks(upper), (1, 1));
        f.sink.removed(f.lower);
        assert_eq!(f.blocks(upper), (0, 0));
        assert_eq!(f.store.cursor(upper, 1), None);
        let rows = f.sink.rows.lock();
        assert!(rows.written.is_empty() && rows.fed.is_empty());
    }

    /// A new connection of the engine's stream is every fed rank's: a
    /// rank's first batch below its cursor then reads as a publisher that
    /// started over.
    #[test]
    fn a_reconnect_reaches_every_rank() {
        let f = Ranked::new();
        let upper = f.upper();
        assert_eq!(
            f.admit(1, 7, Data::Stored(wide_store(1, 16))),
            BatchOutcome::Applied
        );
        f.sink.reconnected(f.lower);
        assert_eq!(
            f.sink.admit(f.lower, 1, facts(6), &batch([])),
            BatchOutcome::Applied
        );
        assert_eq!(f.blocks(upper), (0, 0));
    }

    /// What a rank's batches showed of its groups is its own row's. A group
    /// shown as another kind on rank 1 keeps that group's removal from rank
    /// 1's index, and means nothing for rank 0.
    #[test]
    fn the_admission_state_is_the_routed_rows() {
        let f = Ranked::new();
        let upper = f.upper();
        let shown = batch([
            Data::Stored(main_store(1, 0)),
            Data::Stored(window_store(&[1], 2, false)),
        ]);
        assert_eq!(
            f.sink.admit(f.lower, 1, facts(1), &shown),
            BatchOutcome::Applied
        );
        assert_eq!(
            f.admit(0, 1, Data::Stored(stored(1))),
            BatchOutcome::Applied
        );
        assert_eq!(f.blocks(upper), (1, 1));
        let removal = || Data::Removed(group_removal(&[1], 2));
        // On rank 1 the removal is the window group's: the main block stays.
        assert_eq!(f.admit(1, 2, removal()), BatchOutcome::Applied);
        // Rank 0 showed no group: the same removal is its main block's.
        assert_eq!(f.admit(0, 2, removal()), BatchOutcome::Applied);
        assert_eq!(f.blocks(upper), (0, 1));
    }

    /// A rank's row the stream feeds for the first time begins from
    /// nothing, as a feed's own row does at its open: what the store kept
    /// of it was another writer's.
    #[test]
    fn a_ranks_row_the_stream_feeds_first_begins_from_nothing() {
        let f = Ranked::new();
        let upper = f.upper();
        // Another adapter over the same store wrote the row before.
        let other = KvAdapter::new(Arc::clone(&f.store) as Arc<Store>);
        let before = batch([Data::Stored(wide_store(1, 16))]);
        assert_eq!(
            other.admit(upper, 1, facts(7), &before),
            BatchOutcome::Applied
        );
        assert_eq!(f.blocks(upper), (0, 1));
        // A batch below that cursor is the stream's first for the row.
        assert_eq!(
            f.admit(1, 3, Data::Stored(wide_store(2, 16))),
            BatchOutcome::Applied
        );
        assert_eq!(f.blocks(upper), (0, 1));
        let applied = f.sink.cursor(f.lower, 1).map(|cursor| cursor.applied);
        assert_eq!(applied, Some(3));
    }

    /// A row whose facts move it under another engine is that engine's
    /// stream's from then on. It begins from nothing there, the old
    /// stream's removal leaves it alone, and the new stream's reset reaches
    /// it.
    #[test]
    fn a_row_that_moves_to_another_engine_is_its_new_streams() {
        let f = Ranked::new();
        let upper = f.upper();
        let store = |hash| Data::Stored(wide_store(hash, 16));
        assert_eq!(f.admit(1, 5, store(1)), BatchOutcome::Applied);
        assert_eq!(f.blocks(upper), (0, 1));

        // The same registration names the other engine now: the row keeps
        // its id, and the first engine has no row of rank 1.
        let other = f.other(OTHER);
        let moved = f.store.upsert(rank_facts(UPPER, OTHER, 1)).id();
        assert_eq!(moved, upper);
        f.host.objects.follow(f.store.as_ref());
        assert_eq!(f.admit(1, 6, store(2)), BatchOutcome::Skipped);

        // The other engine's stream takes the row, below its old cursor.
        assert_eq!(
            f.sink.admit(other, 1, facts(2), &batch([store(3)])),
            BatchOutcome::Applied
        );
        assert_eq!(f.store.indexed_blocks(upper, 1), 1);
        let applied = f.sink.cursor(other, 1).map(|cursor| cursor.applied);
        assert_eq!(applied, Some(2));

        f.sink.removed(f.lower);
        assert_eq!(f.store.indexed_blocks(upper, 1), 1);
        f.sink.resync(other, None, ResyncReason::OutOfRange);
        assert_eq!(f.store.indexed_blocks(upper, 1), 0);
    }

    /// A feed's own row whose facts make it a rank of another engine is
    /// that engine's stream's. Its own stream, which is ending, writes no
    /// row of any rank from then on: not its own, and no row of the other
    /// engine by its new facts. That stream's removal leaves what the
    /// other engine's stream wrote.
    #[test]
    fn a_feeds_row_that_becomes_another_engines_rank_leaves_its_own_stream() {
        let f = Ranked::new();
        let own = f.other(OTHER);
        let third = f.store.upsert(rank_facts(THIRD, URL, 2)).id();
        f.host.objects.follow(f.store.as_ref());
        let store = |hash| Data::Stored(wide_store(hash, 16));
        // One batch on the row's own stream.
        let late = |rank, sequence, hash| {
            let one = batch([store(hash)]);
            f.sink.admit(own, rank, facts(sequence), &one)
        };
        assert_eq!(late(0, 5, 1), BatchOutcome::Applied);
        assert_eq!(f.admit(2, 5, store(9)), BatchOutcome::Applied);
        let thirds = || f.store.indexed_blocks(third, 2);
        assert_eq!(thirds(), 1);

        // The same registration states rank 1 of the first engine now. A
        // late batch of the row's own stream reaches no row of that engine,
        // before its stream takes the row and after.
        let moved = f.store.upsert(rank_facts(OTHER, URL, 1)).id();
        assert_eq!(moved, own);
        f.host.objects.follow(f.store.as_ref());
        assert_eq!(late(2, 4, 7), BatchOutcome::Skipped);
        assert_eq!(f.admit(1, 2, store(2)), BatchOutcome::Applied);
        assert_eq!(f.store.indexed_blocks(own, 1), 1);
        assert_eq!(late(0, 6, 3), BatchOutcome::Skipped);
        assert_eq!(late(2, 4, 8), BatchOutcome::Skipped);
        assert_eq!(thirds(), 1);

        // Its facts state rank 0 of its first engine again and no feed was
        // opened for that: the row is still the other stream's, so its old
        // stream writes no rank of its first engine either.
        let fourth = f.store.upsert(rank_facts(FOURTH, OTHER, 1)).id();
        let back = f.store.upsert(rank_facts(OTHER, OTHER, 0)).id();
        assert_eq!(back, own);
        f.host.objects.follow(f.store.as_ref());
        assert_eq!(late(1, 1, 4), BatchOutcome::Skipped);
        assert_eq!(f.store.indexed_blocks(fourth, 1), 0);

        f.sink.removed(own);
        assert_eq!(thirds(), 1);
        f.sink.removed(f.lower);
        assert_eq!(thirds(), 0);
    }

    /// A ticket from before the row's object was laid again opens nothing
    /// and is counted. The row is another stream's by then, and what that
    /// stream wrote stands.
    #[test]
    fn a_ticket_from_before_the_row_was_laid_again_opens_nothing() {
        let f = Ranked::new();
        let row = f.store.upsert(rank_facts(OTHER, OTHER, 0)).id();
        f.host.objects.follow(f.store.as_ref());
        let worker = f
            .host
            .objects
            .registry()
            .get_by_url(OTHER)
            .expect("the other engine's object");
        let ticket = f.host.enroll(&worker, MODEL);
        assert_eq!(ticket, Ok(row));

        // Between the enrollment and the open the row becomes the first
        // engine's rank 1, and that engine's stream writes it.
        let moved = f.store.upsert(rank_facts(OTHER, URL, 1)).id();
        assert_eq!(moved, row);
        f.host.objects.follow(f.store.as_ref());
        let store = Data::Stored(wide_store(1, 16));
        assert_eq!(f.admit(1, 2, store), BatchOutcome::Applied);
        assert_eq!(f.store.indexed_blocks(row, 1), 1);

        assert!(f.host.open(ticket, &worker, MODEL).is_none());
        assert_eq!(f.store.indexed_blocks(row, 1), 1);
        assert_eq!(f.host.skipped_subscriptions().no_row, 1);
    }

    /// No stream takes a row its leaving stream still clears: a batch
    /// routed to it is passed over, and the first one behind the removal
    /// takes it. So a removal clears nothing a new stream wrote.
    #[test]
    fn a_row_its_leaving_stream_still_clears_is_passed_over() {
        let f = Ranked::new();
        let upper = f.upper();
        let store = |hash| Data::Stored(wide_store(hash, 16));
        assert_eq!(f.admit(1, 5, store(1)), BatchOutcome::Applied);
        let other = f.other(OTHER);
        let moved = f.store.upsert(rank_facts(UPPER, OTHER, 1)).id();
        assert_eq!(moved, upper);
        f.host.objects.follow(f.store.as_ref());
        // One batch of rank 1 on the other engine's stream.
        let late = |sequence, hash| {
            let one = batch([store(hash)]);
            f.sink.admit(other, 1, facts(sequence), &one)
        };

        // The first engine's stream is leaving: its removal marked the row
        // and has not cleared it yet.
        f.sink
            .rows
            .lock()
            .written
            .get_mut(&upper)
            .expect("the row's entry")
            .leaving = true;
        assert_eq!(late(2, 2), BatchOutcome::Skipped);
        assert_eq!(f.store.indexed_blocks(upper, 1), 1);

        // The removal ends: the row is cleared and forgotten, and the
        // other engine's next batch takes it.
        f.sink.removed(f.lower);
        assert_eq!(f.store.indexed_blocks(upper, 1), 0);
        assert_eq!(late(3, 3), BatchOutcome::Applied);
        assert_eq!(f.store.indexed_blocks(upper, 1), 1);
        let rows = f.sink.rows.lock();
        let stream = rows.written.get(&upper).map(|written| written.stream);
        assert_eq!(stream, Some(other));
        assert_eq!(rows.fed.get(&other), Some(&vec![upper]));
        assert_eq!(rows.fed.get(&f.lower), None);
    }

    /// A route read before a row was laid again takes no open feed's own
    /// row. The row stands at rank 0 of its own engine and its feed is
    /// open: the old stream's late batch is passed over, and the row's own
    /// stream goes on writing it.
    #[test]
    fn a_stale_route_takes_no_open_feeds_own_row() {
        let f = Ranked::new();
        let upper = f.upper();
        let store = |hash| Data::Stored(wide_store(hash, 16));
        assert_eq!(f.admit(1, 5, store(1)), BatchOutcome::Applied);
        // The row is laid again in place as rank 0 of its own engine, and
        // its feed opens.
        assert_eq!(f.other(UPPER), upper);

        // A batch of the old stream that read its route before the lay.
        assert!(f.sink.applied(f.lower, upper).is_none());
        let own = batch([store(2)]);
        assert_eq!(
            f.sink.admit(upper, 0, facts(1), &own),
            BatchOutcome::Applied
        );
        assert_eq!(f.store.indexed_blocks(upper, 0), 1);
    }

    /// A feed's open asks whether its object still stands for the row in
    /// the hold that writes the row's entry: the lock is taken while the
    /// answer is read.
    #[test]
    fn a_feeds_open_asks_the_holder_in_the_hold_that_writes_its_row() {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let sink = &f.sink;
        let held = sink.begin(f.row, || sink.rows.try_lock().is_none());
        assert_eq!(held, Ok(()));
        {
            let rows = sink.rows.lock();
            let begun = rows.written.get(&f.row).expect("the row's entry");
            assert_eq!((begun.stream, begun.leaving), (f.row, false));
        }
        // A row its object stands for no longer begins nothing.
        sink.removed(f.row);
        assert_eq!(sink.begin(f.row, || false), Err(SkipReason::NoRow));
        assert!(sink.rows.lock().written.is_empty());
    }

    /// A store that says, at each clear of a row, whether the adapter over
    /// it holds its rows.
    struct Clears {
        store: Arc<Store>,
        sink: OnceLock<Weak<KvAdapter<Store>>>,
        held: Mutex<Vec<bool>>,
    }

    impl store::KvEventSink for Clears {
        type Batch = [KvEvent];

        fn admit(
            &self,
            id: WorkerId,
            rank: u16,
            facts: store::BatchFacts,
            batch: &[KvEvent],
        ) -> store::BatchOutcome {
            self.store.admit(id, rank, facts, batch)
        }

        fn admit_through(
            &self,
            id: WorkerId,
            rank: u16,
            facts: store::BatchFacts,
            body: &mut dyn FnMut(&mut LaneWriter<'_>),
        ) -> store::BatchOutcome {
            self.store.admit_through(id, rank, facts, body)
        }

        fn reconnected(&self, id: WorkerId) {
            self.store.reconnected(id);
        }

        fn resync(&self, id: WorkerId, rank: Option<u16>, reason: store::ResyncReason) {
            self.store.resync(id, rank, reason);
        }

        fn cursor(&self, id: WorkerId, rank: u16) -> Option<store::KvCursor> {
            self.store.cursor(id, rank)
        }

        fn removed(&self, id: WorkerId) {
            if let Some(sink) = self.sink.get().and_then(Weak::upgrade) {
                self.held.lock().push(sink.rows.try_lock().is_none());
            }
            self.store.removed(id);
        }

        fn block_size(&self, model: &str) -> Option<u32> {
            self.store.block_size(model)
        }

        fn set_block_size(&self, model: &str, size: u32) {
            self.store.set_block_size(model, size);
        }

        fn forget_block_size(&self, model: &str) {
            self.store.forget_block_size(model);
        }
    }

    /// A row's first clear runs in the hold that writes its entry, at a
    /// stream's take and at a feed's open. No stream claims the row, and
    /// writes it, between the two, so the clear takes nothing a later
    /// stream wrote. A removal's clear runs outside the lock, behind its
    /// mark: it takes every block the stream stored.
    #[test]
    fn a_claimed_rows_first_clear_runs_in_the_hold_that_writes_its_entry() {
        let store = Arc::new(MemoryStore::new());
        let clears = Arc::new(Clears {
            store: Arc::clone(&store) as Arc<Store>,
            sink: OnceLock::new(),
            held: Mutex::default(),
        });
        let sink = Arc::new(KvAdapter::new(Arc::clone(&clears) as Arc<Store>));
        assert!(clears.sink.set(Arc::downgrade(&sink)).is_ok());
        let row = store.upsert(rank_facts(URL, URL, 0)).id();
        let other = store.upsert(rank_facts(OTHER, OTHER, 0)).id();

        // Another stream's take of the row, then the open of its own feed.
        assert!(sink.take(other, row).is_some());
        assert_eq!(sink.begin(row, || true), Ok(()));
        assert_eq!(*clears.held.lock(), [true, true]);

        // The feed's removal.
        sink.removed(row);
        assert_eq!(*clears.held.lock(), [true, true, false]);
    }
}
