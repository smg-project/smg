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
//! row leaves its stream: the stream's last removal is told, another
//! stream takes the row, or a feed's opening releases it. At each of them
//! the state is released: a batch that still holds it stores nothing. The
//! row's reverse map is the lane's: the state holds it for a batch alone,
//! and nothing between batches.
//!
//! The tables empty a rank in their own turn: a resync, a restart or a gap
//! a cursor reads, the worker reported gone. The lane says a value that
//! moves with each emptying, and a batch that reads another value than the
//! row's last one resets the row's state first. A batch locks its row's
//! state ahead of the row's lane and holds it through its body: the state,
//! then the lane. A leaving row's release and the gauge's sum lock a state
//! by itself. A row's move between two streams releases the old stream's
//! state in its hold of the rows' lock, ahead of the row's clear: the
//! rows, then the state, then the lane. Nothing takes the rows' lock while
//! it holds a state or a lane. A take and a feed's opening ask the holder
//! inside the rows' lock, and the holder asks nothing of the adapter.
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
//! One stream writes a row, and it writes only while its feed stands
//! open. The adapter keeps a row's record from a feed's opening until the
//! removal of the row's last open feed is told. A batch on a stream with
//! no record is passed over at every rank: one said behind the removal,
//! or one on a stream no feed was opened for. The worker's next feed
//! opens on the same row and begins it again.
//!
//! A row named under another address has two feeds open for a time, and
//! the record counts them: a removal that leaves a feed open takes
//! nothing. A row of another rank that the row's stream wrote under the
//! first address, and that stays there, is no rank's row of the row's
//! engine now, and the row's stream reaches it no longer. A feed's
//! opening asks the holder for the rows of its row's engine and releases
//! every other row its stream still writes: each is cleared and
//! forgotten, as at a removal. A row another stream took since is that
//! stream's, and stays.
//!
//! A row whose facts move it under another
//! engine is that engine's stream's from its next batch: it begins from
//! nothing there, and the stream that wrote it reaches it no longer. A
//! feed's own row that becomes a rank of another engine is taken the same
//! way, and its own stream, which is ending, leaves it alone. That stream
//! writes no row of any rank until the stream that took its row has left,
//! and the holder answers it no rank once its row stands above rank 0. A
//! ticket from before a row's object was laid again opens no feed.
//!
//! A row's stream and a stream's rows are kept under one lock: a row
//! moves between two streams in one step, and its lane is cleared in
//! that step, so the clear takes nothing a later stream wrote. The state
//! its old stream wrote it under is released in that step, ahead of the
//! clear: a batch of that stream that still holds it stores nothing
//! behind the move. A stream that leaves reaches only the rows it still
//! writes. It marks them, clears them in the tables outside that lock,
//! which every row's batch takes, and forgets their entries once those
//! clears have ended: each stays its row while its mark stands. A stream
//! that would take one in between passes its batch over, so a removal
//! clears nothing a new stream wrote. The limits name an exception: two
//! removals of one row that overlap. The state the adapter kept of each
//! leaving row is released ahead of its clear, and the state its mark
//! bears is released from the first. A batch reads the release ahead of
//! the row's cursor and holds the state through its admission: that runs
//! ahead of the clear, which takes what it stored, or the batch is
//! passed over with the cursor where it was.
//!
//! A stream takes a row of another rank only where the holder routes
//! that rank's batch to the row in the hold that writes the row's entry.
//! A batch reads its route ahead of its feed's record, and a take reads
//! it again in that hold: where the holder gives the route no longer,
//! the batch takes none and is passed over. So, by the holder's answer
//! at that read, an open feed's own row that stands at rank 0 is taken
//! by no other stream, and a row an opening released is not taken back
//! by a route the cut took away. The limits name three things that read
//! leaves: a holder that moves the row behind it, a route the holder
//! gives again, and two rows at rank 0 that dial one engine address. A
//! feed's open asks the holder whether its object still stands for the
//! row, and whether the row still stands at rank 0, in the hold that
//! writes the row's entry. An object the holder kept through a move
//! stands for its row above rank 0 too, so the first answer does not
//! give the second. A row laid again behind those answers is taken from
//! the stale feed by the stream the holder then routes to it, and by no
//! batch of the stale feed from that stream.
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
//!   written by its old stream until that stream's removal is told. The
//!   rows of other ranks that stream still writes stand until the new
//!   engine's feed opens on the row, or until the row's last removal.
//! - A take reads the holder's route once, in its hold. A row the holder
//!   moves behind that read is taken all the same, and its lane is
//!   cleared. It is the stream's until the stream the holder then routes
//!   its rank to takes it, a later opening releases or takes it, or the
//!   stream's last removal. Where the holder stood it at rank 0 again
//!   for an open feed of its own, that feed's batches are passed over
//!   until then.
//! - A reset or a new connection a row's old stream had in flight as the
//!   row moved can land behind the new stream's clear.
//! - What a lane's cursor decides by itself is in no series: a gap kept
//!   or cleared, the batches missed, the degraded ranks, and the reason it
//!   emptied a rank ahead of a batch.
//! - The index-blocks gauge moves at a batch: a rank the tables emptied
//!   shows at its row's next batch.
//! - A batch the adapter passes over by itself is counted nowhere: one of
//!   a rank with no row laid, one that would take a row the holder routes
//!   it to no longer, one on a stream that is ending, one on a stream
//!   with no open feed, or one that holds a released state.
//! - A batch that finds a row's mark where it would take the row, or
//!   that gets the mark's state, is passed over, and no replay is asked:
//!   another stream's first batch for the row, and a batch of the row's
//!   own next feed, which opens on the mark.
//! - A batch in flight as its stream's removal is told has read its
//!   feed's record and goes on. Under a state the adapter kept of its
//!   row its admission runs ahead of the removal's clear, or it is
//!   passed over. A state made for the batch itself is not released, and
//!   such a batch can store behind the clear. The stream's own row gets
//!   one where no entry is kept of it, and the adapter keeps it behind
//!   an applied batch where the row has no entry then. A row of another
//!   rank can be taken anew where the removal did not list it, or has
//!   ended, and the holder still routes the batch's rank to it. What
//!   such a batch stored can stand through the worker's next
//!   feed's opening. An opening clears its row only where it claims it,
//!   and it claims no row that bears a mark or an entry of its own
//!   stream. It releases a row of another rank that the holder's answer
//!   does not name.
//! - An opening holds each row of another rank its stream writes against
//!   the holder's answer under the rows' lock: a search of the engine's
//!   rows for each. It reads that answer once. Where the feed's row is
//!   laid again behind it, the opening can release a row its stream
//!   reaches again: that row begins from nothing where the stream takes
//!   it again.
//! - Two last removals of one row can overlap: a feed opens on the
//!   first's mark and ends ahead of the first's end. Each forgets the
//!   row's entry by its key, so the row can stand unmarked while one of
//!   them has not cleared it. That clear can take what a later feed
//!   stored, and that removal's end can forget the later feed's entry.
//! - While a row has two feeds open, a batch of the first address's
//!   stream is the row's stream's as the second's is, at every rank: the
//!   same rules route it and write it. One that read its route ahead of
//!   the cut can find that route's row the stream's again, where the
//!   holder gives the route again. Where the batch holds the row's state
//!   from ahead of the release, it is passed over. Where it gets the
//!   row's state behind the new claim, the row's cursor reads its
//!   sequence as the stream's: a restart it reads there empties the row.
//! - Two rows at rank 0 that dial one engine address are outside the
//!   layout the feed is written for. Both streams route the address's
//!   rows of other ranks: each takes such a row from the other at its
//!   batch, with a clear, and a batch that holds the row's state from
//!   ahead of that take is passed over.
//! - An opening whose task ends with no removal leaves its row counted,
//!   and the row's later removals take nothing.

use std::{
    collections::HashMap,
    mem,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
};

use parking_lot::Mutex;
use smg_worker::{
    common_proto::{kv_cache_event::Data, KvEventBatch},
    identity::basic::Worker,
    kv_events::{
        apply::{apply_counted, BlockIndex, HeldBlocks, WorkerIndexState},
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
    /// What the streams said of their models' block sizes: the widest
    /// block each model's stored blocks named, the engine's word, which
    /// only rises. A learning and a model's departure hold it through
    /// their calls of the tables.
    sizes: Mutex<HashMap<Arc<str>, u32>>,
    /// What each row's open feeds were opened with, by the feeds' own row:
    /// one record a row, which counts them. Taken alone, or under `sizes`
    /// by a learning that asks whether its feed still stands: never ahead
    /// of `sizes`.
    feeds: Mutex<HashMap<WorkerId, Arc<Feed>>>,
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
    /// Write `stream` as `row`'s stream, from nothing shown, for a take
    /// and for a feed's opening alike: the row leaves the list of the
    /// stream that wrote it, and the state that stream wrote it under is
    /// released ([`Applied::release`]). The caller holds the rows' lock and
    /// clears the row's lane next. The release waits there for a batch of
    /// the old stream that is inside its admission under that state, as
    /// that clear waits for the batch's lane. The row's new state comes
    /// back. `None`, with nothing done, where the row is leaving that
    /// stream and its mark still stands.
    fn claim(&mut self, stream: WorkerId, row: WorkerId) -> Option<Arc<Mutex<Applied>>> {
        if self.written.get(&row).is_some_and(|before| before.leaving) {
            return None;
        }
        let begun = Written::begun(stream);
        let applied = Arc::clone(&begun.applied);
        if let Some(before) = self.written.insert(row, begun) {
            Applied::release(&before.applied);
            if let Some(list) = self.fed.get_mut(&before.stream) {
                list.retain(|&other| other != row);
            }
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
    /// lock: a batch holds it from ahead of the row's cursor through its
    /// body ([`KvAdapter::admit_under`]).
    applied: Arc<Mutex<Applied>>,
    /// The stream that writes the row, named by its own row: the row
    /// itself for a feed's row, its engine's rank 0 row for the row of
    /// another rank.
    stream: WorkerId,
    /// The row is leaving its stream: no stream takes the row while this
    /// mark stands. The stream's last removal marks every row it writes
    /// so, and a feed's opening the rows of other ranks its stream reaches
    /// no longer. Each forgets its rows' entries once their clears have
    /// ended ([`KvAdapter::leave`]).
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

    /// The mark of a row that is leaving `stream`. Its state is released
    /// from its making: a batch of the stream that reads the mark stores
    /// nothing.
    fn mark(stream: WorkerId) -> Self {
        let released = Applied {
            released: true,
            ..Applied::default()
        };
        Self {
            applied: Arc::new(Mutex::new(released)),
            stream,
            leaving: true,
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
    /// The row left the stream this state was written under: at a removal
    /// or an opening's release ([`KvAdapter::leave`]), or at its move to
    /// another stream ([`Rows::claim`]). A batch that holds the state
    /// stores nothing under it from then on.
    released: bool,
}

impl Applied {
    /// The row left the stream `state` was written under. The one routine
    /// that says so of a state that stands: a mark's state is made
    /// released ([`Written::mark`]). A batch inside its admission under
    /// the state is waited for, and that admission runs first. One that
    /// locks the state behind this reads the release.
    fn release(state: &Mutex<Self>) {
        state.lock().released = true;
    }
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

/// The lane's side of a batch's loan: the lane's map, out of the lane for
/// one batch. The row's state holds it for the rules' call alone and hands
/// it back here ([`WorkerIndexState::holding`]). Dropped, this puts the
/// map back in the lane, also behind rules that unwound: the tables then
/// empty the rank by a map that names every block the row holds in the
/// chain. The map names nothing of the row's group view: the tables clear
/// the view in a step of its own.
struct Lending<'a, 'w> {
    lane: &'a mut LaneWriter<'w>,
    map: Lent,
}

impl<'a, 'w> Lending<'a, 'w> {
    fn new(lane: &'a mut LaneWriter<'w>) -> Self {
        let map = Lent(Some(mem::take(lane.map())));
        Self { lane, map }
    }
}

impl Drop for Lending<'_, '_> {
    fn drop(&mut self) {
        if let Some(map) = self.map.0.take() {
            *self.lane.map() = map;
        }
    }
}

/// Where the ranks of an engine stand: the holder's answer, as the adapter
/// reads it to route a batch.
trait Ranks: Send + Sync {
    /// The row of rank `rank` of the engine `row` is a rank of.
    fn row_of(&self, row: WorkerId, rank: u16) -> Option<WorkerId>;

    /// The rows of every rank of the engine whose rank 0 row is `row`,
    /// its own among them: none where `row` stands at rank 0 no longer.
    fn rows_of(&self, row: WorkerId) -> Vec<WorkerId>;
}

impl<G: Generation> Ranks for Objects<G> {
    fn row_of(&self, row: WorkerId, rank: u16) -> Option<WorkerId> {
        self.rank_row(row, rank)
    }

    fn rows_of(&self, row: WorkerId) -> Vec<WorkerId> {
        self.rank_rows(row)
    }
}

/// What a row's feed was opened with: one record a row, kept from a
/// feed's opening until the removal of the row's last open feed is told.
/// A stream writes only while its record stands.
///
/// A row can have two feeds open for a time. A cut that names a standing
/// row under another address ends the object at the first address and
/// lays one at the second, and a collector orders nothing across two
/// addresses. So the record counts its row's open feeds. An opening
/// counts one more and writes the record's words anew, and a removal
/// counts one less. A removal that leaves a feed open takes nothing: no
/// record, no row and no mark. The removal of the last takes the record
/// and the rows.
///
/// An opening releases the rows of other ranks that the row's stream
/// still writes and reaches no longer ([`KvAdapter::begin`]): behind
/// such a cut the holder names them as no rank's row of the row's
/// engine. Two feeds of one address keep their rank rows.
struct Feed {
    /// The model the feed's rows stream for: their stored blocks name its
    /// block size.
    model: Arc<str>,
    /// The URL the feed's worker is named by in the cache-event series.
    url: Box<str>,
    /// The row's open feeds, this record's own among them. Read and
    /// written under the lock of the adapter's records alone.
    open: AtomicU32,
}

impl KvAdapter {
    fn over(tables: Arc<KvTables>, ranks: Option<Arc<dyn Ranks>>) -> Self {
        Self {
            rows: Mutex::new(Rows::default()),
            sizes: Mutex::new(HashMap::new()),
            feeds: Mutex::new(HashMap::new()),
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
    /// table keeps it. A row that is leaving the stream answers its mark's
    /// state, which is released ([`Written::mark`]). The stream's own row
    /// that no entry is kept of gets a state the table takes once a batch
    /// was admitted ([`Self::keep`]).
    /// A removal forgot that entry while the stream's record stood, or
    /// while a batch that had read it was in flight. The row's feed opened
    /// while a leaving stream still cleared the row, and that removal has
    /// ended ([`Self::begin`]). Or the batch read its feed's record ahead
    /// of its own stream's removal and comes here behind that removal's
    /// end. Or another stream took the row, and that stream's removal has
    /// ended while the row's own is not told yet.
    /// `None` where the stream writes no row any longer: another stream
    /// took its own row, so it is ending, and it writes no row of any
    /// rank. A row of another rank that the stream writes for the first
    /// time, or that another stream wrote, is taken for the batch's rank
    /// ([`Self::take`]): `None` where it is not the stream's to take.
    fn applied(
        &self,
        id: WorkerId,
        (row, rank): (WorkerId, u16),
    ) -> Option<(Arc<Mutex<Applied>>, bool)> {
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
            // The stream's own row that no entry is kept of. Its feed
            // opened while a leaving stream still cleared the row, or the
            // batch read its record ahead of its own stream's removal, or
            // the stream that took the row has left.
            None if row == id => Some((Arc::default(), false)),
            _ => self.take(id, (row, rank)).map(|applied| (applied, true)),
        }
    }

    /// `stream` takes `row`, the row of `rank`, for a batch of that rank:
    /// another rank than the stream's own row's. The row begins from
    /// nothing, as a feed's own row does at its open: what the tables kept
    /// of it was decided by facts that are gone, or by another stream's
    /// batches. The stream that wrote it reaches it no longer: the row
    /// changes lists, and its lane is cleared, in one hold of the lock. The
    /// clear is of one row, and of nothing where the tables never held it.
    /// The row's new state comes back. `None`, with nothing done, where the
    /// row is not this stream's to take, and the batch is passed over. The
    /// holder routes the stream's batch of `rank` to it no longer: the
    /// route is read again in the hold, so one the holder took back since
    /// the batch read it takes no row. That read is one, and the module's
    /// limits name a row the holder moves behind it. Or the row is leaving
    /// another stream and its mark still stands: a later batch can take
    /// the row.
    ///
    /// A row that is the stream's by this hold hands the state it has, and
    /// nothing is cleared again: the adapter keeps one state of a row. A
    /// batch of the stream's other feed took the row since this batch
    /// asked for its state. Or the stream's removal, or its feed's
    /// opening, marked the row since: the state is the mark's, which is
    /// released, and the batch is passed over where it reads the release.
    fn take(&self, stream: WorkerId, (row, rank): (WorkerId, u16)) -> Option<Arc<Mutex<Applied>>> {
        let mut rows = self.rows.lock();
        let before = rows.written.get(&row);
        if let Some(taken) = before.filter(|before| before.stream == stream) {
            return Some(Arc::clone(&taken.applied));
        }
        // The holder is asked again, in the hold: a route it gives no
        // longer takes none.
        if self.routed(stream, rank) != Some(row) {
            return None;
        }
        let applied = rows.claim(stream, row)?;
        // In the hold that wrote the entry: no stream claims the row, and
        // writes it, ahead of this clear.
        self.tables.removed(row);
        Some(applied)
    }

    /// Rows leave `stream`, in this order. `named` lists them under the
    /// rows' lock, and each is marked in that one hold, the state that
    /// stood under it kept aside. The lock is let go. Each state kept
    /// aside is released under its own lock, with no other lock held.
    /// Each row is cleared in the tables. Each row's entry is forgotten,
    /// by its key. A stream that would take a row in between passes its
    /// batch over, so nothing a new stream wrote is cleared here: the
    /// module's limits name an exception, two removals of one row that
    /// overlap.
    ///
    /// The release stands ahead of the clear. A batch locks its row's
    /// state, reads the release and holds the state through its admission
    /// ([`Self::admit_under`]): that runs ahead of the release, and the
    /// clear takes what it stored, or the batch is passed over. A release
    /// waits for such a batch's admission and a clear for the lane's turn,
    /// and the rows' lock is held for neither. The clears take every block the
    /// stream stored under the states set aside here: the module's limits
    /// name the batch that holds another. What `named` answers beside its
    /// rows comes back.
    fn leave<T>(&self, stream: WorkerId, named: impl FnOnce(&mut Rows) -> (Vec<WorkerId>, T)) -> T {
        let (leaving, stood, answer) = {
            let mut rows = self.rows.lock();
            let (leaving, answer) = named(&mut rows);
            let mut stood = Vec::with_capacity(leaving.len());
            for &row in &leaving {
                let before = rows.written.insert(row, Written::mark(stream));
                stood.extend(before.map(|before| before.applied));
            }
            (leaving, stood, answer)
        };
        for state in &stood {
            Applied::release(state);
        }
        for &row in &leaving {
            self.tables.removed(row);
        }
        let mut rows = self.rows.lock();
        for row in &leaving {
            rows.written.remove(row);
        }
        answer
    }

    /// The rows of other ranks that `row`'s stream writes and reaches no
    /// longer, taken out of its list: `reached`, the holder's answer for
    /// `row`'s engine, names none of them.
    fn unreached(rows: &mut Rows, row: WorkerId, reached: &[WorkerId]) -> Vec<WorkerId> {
        let Some(fed) = rows.fed.get_mut(&row) else {
            return Vec::new();
        };
        let (kept, released) = mem::take(fed)
            .into_iter()
            .partition(|other| reached.contains(other));
        *fed = kept;
        released
    }

    /// A feed's own row begins. An error, with nothing done, says why it
    /// does not. [`SkipReason::NoRow`]: `stands` says the feed's object stands
    /// for the row no longer. [`SkipReason::UpperRank`]: the holder stands the
    /// row above rank 0. An object can stand for such a row and state no
    /// rank: it states one where its row's width is above one. Both are asked in
    /// the hold that writes the row's entry: a row laid again behind the
    /// answers is taken from this feed by the stream the holder then
    /// routes to it, and never by this feed from that stream. The holder's
    /// ranks are asked once at most: the rows of the row's engine, none
    /// where the row stands at rank 0 no longer, and the same answer
    /// decides what the opening releases.
    ///
    /// A row the adapter holds no facts for begins from nothing: what the
    /// tables kept of its stream was decided by facts that are gone. The
    /// row's lane starts over at whatever rank it stands, and a row with
    /// none loses nothing. A row another stream wrote, as a rank of
    /// its engine, begins from nothing too, and the state that stream
    /// wrote it under is released ([`Rows::claim`]). One a leaving stream
    /// still clears is left to it, and this feed's batches are passed over
    /// until that ends. Its first batch behind that end begins the row
    /// ([`Self::applied`]).
    ///
    /// The rows of other ranks the row's stream still writes are held
    /// against the holder's answer in the same hold. One it names as no
    /// rank's row of the row's engine is released: it leaves the stream as
    /// a row does at a removal ([`Self::leave`]). Two feeds of one address
    /// keep their rank rows.
    fn begin(&self, row: WorkerId, stands: impl FnOnce() -> bool) -> Result<(), SkipReason> {
        self.leave(row, |rows| {
            if !stands() {
                return (Vec::new(), Err(SkipReason::NoRow));
            }
            // An adapter with no holder to ask routes every batch to its
            // stream's own row: it writes no row of another rank.
            let reached = self.ranks.as_ref().map(|ranks| ranks.rows_of(row));
            if reached.as_ref().is_some_and(Vec::is_empty) {
                return (Vec::new(), Err(SkipReason::UpperRank));
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
            let reached = reached.unwrap_or_default();
            (Self::unreached(rows, row, &reached), Ok(()))
        })
    }

    /// Keep the state `id`'s own row was applied under, where the table
    /// kept none of the row as its batch began. The row's feed opened
    /// while a leaving stream still cleared it. Or the batch read its
    /// feed's record ahead of its own stream's removal and landed behind
    /// that removal's end. Or the stream that took the row has left, and
    /// the row's own stream's removal is not told yet. An entry made in
    /// between is left: another stream took the row behind the batch's
    /// read.
    fn keep(&self, id: WorkerId, applied: Arc<Mutex<Applied>>) {
        let mut rows = self.rows.lock();
        rows.written.entry(id).or_insert(Written {
            applied,
            stream: id,
            leaving: false,
        });
    }

    /// A feed opened for `row`, its own row: its stored blocks name
    /// `model`'s block size, and its series go under `url`, its worker's.
    /// The opening is counted for the row, and the feed's removal takes
    /// the count back ([`Feed`]). An opening whose task ends with no
    /// removal leaves its row counted, and the row's later removals take
    /// nothing: a collector that tells none, or a host kept across two
    /// collectors.
    fn opened(&self, row: WorkerId, model: &str, url: &str) {
        let mut feeds = self.feeds.lock();
        let open = feeds
            .get(&row)
            .map_or(0, |before| before.open.load(Ordering::Relaxed));
        let feed = Feed {
            model: Arc::from(model),
            url: Box::from(url),
            open: AtomicU32::new(open + 1),
        };
        feeds.insert(row, Arc::new(feed));
    }

    /// The last events subscription of `model` ended: the engine's word
    /// for it goes, and the tables' size of record with it, so a model
    /// that returns names its size anew. Both go in one hold of the lock a
    /// learning holds: a row that learns beside the departure writes ahead
    /// of both removals or behind both, and behind them only where its
    /// feed still stands ([`Self::learn`]).
    fn model_left(&self, model: &str) {
        let mut sizes = self.sizes.lock();
        sizes.remove(model);
        self.tables.forget_block_size(model);
    }

    /// A row of `feed`, the feed its batch found open for `row`, holds a
    /// block of record of `size` tokens: what it says of the model's block
    /// size, by the module's rule. A feed that is no longer `row`'s names
    /// no size: its stream left behind the batch's start, and its model's
    /// departure may have cleared the size since.
    fn learn(&self, row: WorkerId, feed: &Arc<Feed>, size: u32) {
        // Held through the tables' calls: two rows of one model learn in turn.
        let mut sizes = self.sizes.lock();
        // Read under that hold, so a departure behind the stream's removal
        // finds this write done or never made.
        let stands = |held: &Arc<Feed>| Arc::ptr_eq(held, feed);
        if !self.feeds.lock().get(&row).is_some_and(stands) {
            return;
        }
        let model = &feed.model;
        let last = sizes.get(model).copied();
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
            sizes.insert(Arc::clone(model), word);
        }
    }

    /// One batch of `id`'s stream to `row`, the row of `rank`, under the
    /// state the stream got for it: `applied`, and whether the table
    /// keeps it. What an admission does behind its reads of the batch's
    /// route, its feed and its row's state ([`KvEventSink::admit`]).
    ///
    /// The state is locked ahead of the tables and held through the
    /// admission. A released state is read there, ahead of the row's
    /// cursor: its row left the stream, or is leaving it ([`Self::leave`]).
    /// Such a batch is passed over with the cursor where it was, and
    /// counted nowhere.
    fn admit_under(
        &self,
        id: WorkerId,
        (row, rank): (WorkerId, u16),
        feed: &Arc<Feed>,
        (applied, kept): (Arc<Mutex<Applied>>, bool),
        facts: BatchFacts,
        batch: &KvEventBatch,
    ) -> BatchOutcome {
        let url = &*feed.url;
        let snapshot = facts.snapshot.is_some();
        // A batch that carries a store names the model's block size.
        let stores = batch
            .events
            .iter()
            .any(|event| matches!(event.data, Some(Data::Stored(_))));
        let mut record = None;
        let mut took = None;
        // Held from this read through the admission: a release behind the
        // read waits for it, and its clear takes what the batch stored.
        let mut locked = applied.lock();
        if locked.released {
            return BatchOutcome::Skipped;
        }
        // The cursor reads the header alone. The body runs where it admits
        // the batch: inside the lane's turn, with the table's lock let go.
        let outcome = self.tables.admit_through(row, rank, facts, &mut |lane| {
            let Applied {
                state,
                emptied,
                blocks,
                ..
            } = &mut *locked;
            if *emptied != Some(lane.emptied()) {
                // The tables emptied the rank since the row's last batch, or
                // the row begins: the copies, the block of record and the
                // view's blocks go with the row's blocks. The view is the
                // index's, under the row's id there. A lane made again holds
                // another: a state that holds a view takes that one, and the
                // rules write it.
                state.index_emptied_in(&Lane(lane), lane.worker());
            }
            let worker = lane.worker();
            {
                let mut lending = Lending::new(lane);
                let (op, disposition) = if snapshot {
                    ("snapshot", "snapshot")
                } else {
                    ("stored", "applied")
                };
                // Applied and accounted for under the worker's name, with
                // the lane's map the state's for the call.
                let (spent, count) = state.holding(&mut lending.map, |state| {
                    apply_counted(
                        batch,
                        url,
                        worker,
                        &Lane(lending.lane),
                        state,
                        op,
                        disposition,
                    )
                });
                took = Some(spent);
                *blocks = count;
            }
            *emptied = Some(lane.emptied());
            record = state.block_of_record();
        });
        // Let go ahead of the gauge's sum, which locks each state.
        drop(locked);
        match outcome {
            BatchOutcome::Applied => {
                if !kept {
                    self.keep(id, applied);
                }
                // A batch whose feed left since it began names no model.
                if let Some(width) = record.filter(|_| stores) {
                    self.learn(id, feed, width);
                }
                // A live batch's duration is the stream's pace. A snapshot
                // chunk publishes none.
                if let Some(took) = took.filter(|_| !snapshot) {
                    Metrics::record_kv_event_apply(url, took.as_secs_f64());
                }
                Metrics::set_kv_index_blocks(url, self.held(id));
            }
            BatchOutcome::Skipped => {
                // A batch behind its lane's cursor. A row with no lane
                // skips every batch and counts none.
                if self.tables.cursor(row, rank).is_some() {
                    Metrics::record_kv_event_batch(url, "stale");
                }
            }
            BatchOutcome::Gap { .. } => {
                // The collector asks the publisher to replay from here.
                Metrics::record_kv_event_gap(url, "replay_requested", 0);
            }
        }
        outcome
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
        // The stream's feed: its worker as its series name it, and the
        // model its stores name a size for. A stream writes only while
        // its record stands: none does where no feed was opened for it,
        // or where its removal was told. Asked ahead of the row's state,
        // so such a batch takes no row either.
        let Some(feed) = self.feeds.lock().get(&id).cloned() else {
            return BatchOutcome::Skipped;
        };
        // No state: another stream took this stream's own row, so it
        // writes none now, or the row is not the stream's to take.
        let Some(state) = self.applied(id, (row, rank)) else {
            return BatchOutcome::Skipped;
        };
        self.admit_under(id, (row, rank), &feed, state, facts, batch)
    }

    fn reconnected(&self, id: WorkerId) {
        for row in self.written_by(id) {
            self.tables.reconnected(row);
        }
    }

    fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        if let Some(feed) = self.feeds.lock().get(&id) {
            Metrics::record_kv_event_resync(&feed.url, reason.as_str());
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
        {
            let mut feeds = self.feeds.lock();
            // One feed of the row ended. Where another stays open, this
            // removal takes nothing: the record counts the row's feeds.
            let stays = |feed: &Arc<Feed>| feed.open.fetch_sub(1, Ordering::Relaxed) > 1;
            if feeds.get(&id).is_some_and(stays) {
                return;
            }
            // The record goes first: a batch that begins behind this step
            // is passed over.
            feeds.remove(&id);
        }
        // What the stream still writes goes with it, since no stream feeds
        // it now: the rows of other ranks, and its own unless another
        // stream took it.
        self.leave(id, |rows| {
            let own = rows.written.get(&id).is_none_or(|own| own.stream == id);
            let mut leaving = rows.fed.remove(&id).unwrap_or_default();
            if own {
                leaving.push(id);
            }
            (leaving, ())
        });
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
                self.sink.opened(row, model, worker.url());
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
        self.sink.model_left(model);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/kv/tests/mod.rs"]
mod tests;
