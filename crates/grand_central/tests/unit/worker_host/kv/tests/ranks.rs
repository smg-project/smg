//! The ranks of an engine: the row each rank's batch is written to, and
//! the rows its stream's removal, resync and new connection reach. A row
//! that moves between two streams, and what the stream it leaves still
//! reaches.

use super::*;

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

/// An engine of three ranks: its rank 0 row, the host whose holder
/// writes the store, and the feed the host opened for that row.
struct Ranked {
    store: Arc<MemoryStore>,
    /// The tables the holder was handed: it makes each row's lane.
    tables: Arc<KvTables>,
    host: KvHost<Members>,
    sink: Arc<KvAdapter>,
    lower: WorkerId,
}

impl Ranked {
    /// Rank 0's row alone: rank 1's is laid by [`Self::upper`].
    fn new() -> Self {
        let store = Arc::new(MemoryStore::new());
        let tables = Arc::new(KvTables::default());
        let objects = Arc::new(
            Objects::new(Arc::new(WorkerRegistry::new())).with_kv_tables(Arc::clone(&tables)),
        );
        let lower = objects.upsert(&store, rank_facts(URL, URL, 0), Registrant::Control);
        let lower = lower.expect("the store takes the write").id();
        let host = KvHost::new(objects);
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
            tables,
            host,
            sink,
            lower,
        }
    }

    /// Write the row `facts` state through the host's holder: its id.
    fn write(&self, facts: WorkerFacts) -> WorkerId {
        let objects = &self.host.objects;
        let written = objects.upsert(&self.store, facts, Registrant::Control);
        written.expect("the store takes the write").id()
    }

    /// Rank 1's row, laid where it is not yet.
    fn upper(&self) -> WorkerId {
        self.write(rank_facts(UPPER, URL, 1))
    }

    /// The rank 0 row of another engine at `key`, laid, with its feed
    /// opened by the host.
    fn other(&self, key: &str) -> WorkerId {
        let row = self.write(rank_facts(key, key, 0));
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
            self.tables.indexed_blocks(self.lower, 0),
            self.tables.indexed_blocks(upper, 1),
        )
    }

    /// A cut names the engine's rank 0 row under `address`: its object at
    /// the first address ends, and one is laid at `address`.
    fn cut_to(&self, address: &str) {
        let mut cut = self.store.checkpoint();
        let moved = cut.rows.iter_mut().find(|row| row.id == self.lower);
        moved.expect("the row").facts = rank_facts(address, address, 0);
        assert!(self.host.objects.watch_from(&self.store, cut));
    }

    /// The object at `address` opens a feed on the engine's rank 0 row:
    /// one more, beside the feeds that stand open for the row.
    fn open_at(&self, address: &str) {
        let object = self.host.objects.registry().get_by_url(address);
        let object = object.expect("the object at the address");
        let ticket = self.host.enroll(&object, MODEL);
        assert_eq!(ticket, Ok(self.lower));
        assert!(self.host.open(ticket, &object, MODEL).is_some());
    }

    /// The record of the engine's feed: what a batch on its stream reads
    /// ahead of its row's state.
    fn feed(&self) -> Arc<Feed> {
        let feed = self.sink.feeds.lock().get(&self.lower).cloned();
        feed.expect("the stream's feed")
    }
}

/// A batch of rank 1 on the engine's one stream is indexed under rank
/// 1's row, and a batch of rank 0 under the stream's own. A rank's store
/// names the model's block size by its stream's feed, which is opened
/// for the stream's own row.
#[test]
fn a_batch_of_rank_one_is_indexed_under_rank_ones_row() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.tables.block_size(MODEL), None);
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
    assert_eq!(f.blocks(upper), (0, 1));
    assert_eq!(f.admit(0, 1, store(2)), BatchOutcome::Applied);
    assert_eq!(f.admit(0, 2, store(3)), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (2, 1));
}

/// A stream that left takes its own feed's record alone: another
/// engine's stream still names the model's block size by its own.
#[test]
fn a_stream_that_left_takes_its_own_feed_alone() {
    let f = Ranked::new();
    let other = f.other(OTHER);
    f.sink.removed(f.lower);
    let first = f.sink.admit(other, 0, facts(1), &wide(1, 16));
    assert_eq!(first, BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
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
    assert_eq!(f.tables.resyncs(ResyncReason::GapCleared), 1);

    assert_eq!(f.admit(1, 8, store(3)), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (1, 1));
    f.sink.resync(f.lower, None, ResyncReason::OutOfRange);
    assert_eq!(f.blocks(upper), (0, 0));
    assert_eq!(f.tables.resyncs(ResyncReason::OutOfRange), 2);
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
    assert_eq!(f.tables.cursor(upper, 1), None);
    let rows = f.sink.rows.lock();
    assert!(rows.written.is_empty() && rows.fed.is_empty());
}

/// Behind its engine's removal a batch is passed over at every rank,
/// another rank's first and then the stream's own. Neither writes: no
/// block, no cursor, and no row taken for the stream.
#[test]
fn a_batch_behind_its_engines_removal_is_passed_over_at_every_rank() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(0, 1, store(1)), BatchOutcome::Applied);
    assert_eq!(f.admit(1, 1, store(2)), BatchOutcome::Applied);
    f.sink.removed(f.lower);
    for rank in [1, 0] {
        assert_eq!(f.admit(rank, 2, store(3)), BatchOutcome::Skipped);
        assert_eq!(f.blocks(upper), (0, 0), "rank {rank}");
        assert_eq!(f.sink.cursor(f.lower, rank), None, "rank {rank}");
        let rows = f.sink.rows.lock();
        let nothing = rows.written.is_empty() && rows.fed.is_empty();
        assert!(nothing, "rank {rank}: a row is kept for the stream");
    }
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
/// nothing, as a feed's own row does at its open: what the tables kept
/// of it was another writer's.
#[test]
fn a_ranks_row_the_stream_feeds_first_begins_from_nothing() {
    let f = Ranked::new();
    let upper = f.upper();
    // Another adapter over the same store wrote the row before, as a
    // feed of the row's own.
    let other = KvAdapter::new(Arc::clone(&f.tables));
    open_feed(&other, upper);
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
    let moved = f.write(rank_facts(UPPER, OTHER, 1));
    assert_eq!(moved, upper);
    assert_eq!(f.admit(1, 6, store(2)), BatchOutcome::Skipped);

    // The other engine's stream takes the row, below its old cursor.
    assert_eq!(
        f.sink.admit(other, 1, facts(2), &batch([store(3)])),
        BatchOutcome::Applied
    );
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    let applied = f.sink.cursor(other, 1).map(|cursor| cursor.applied);
    assert_eq!(applied, Some(2));

    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    f.sink.resync(other, None, ResyncReason::OutOfRange);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
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
    let third = f.write(rank_facts(THIRD, URL, 2));
    let store = |hash| Data::Stored(wide_store(hash, 16));
    // One batch on the row's own stream.
    let late = |rank, sequence, hash| {
        let one = batch([store(hash)]);
        f.sink.admit(own, rank, facts(sequence), &one)
    };
    assert_eq!(late(0, 5, 1), BatchOutcome::Applied);
    assert_eq!(f.admit(2, 5, store(9)), BatchOutcome::Applied);
    let thirds = || f.tables.indexed_blocks(third, 2);
    assert_eq!(thirds(), 1);

    // The same registration states rank 1 of the first engine now. A
    // late batch of the row's own stream reaches no row of that engine,
    // before its stream takes the row and after.
    let moved = f.write(rank_facts(OTHER, URL, 1));
    assert_eq!(moved, own);
    assert_eq!(late(2, 4, 7), BatchOutcome::Skipped);
    assert_eq!(f.admit(1, 2, store(2)), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(own, 1), 1);
    assert_eq!(late(0, 6, 3), BatchOutcome::Skipped);
    assert_eq!(late(2, 4, 8), BatchOutcome::Skipped);
    assert_eq!(thirds(), 1);

    // Its facts state rank 0 of its first engine again and no feed was
    // opened for that: the row is still the other stream's, so its old
    // stream writes no rank of its first engine either.
    let fourth = f.write(rank_facts(FOURTH, OTHER, 1));
    let back = f.write(rank_facts(OTHER, OTHER, 0));
    assert_eq!(back, own);
    assert_eq!(late(1, 1, 4), BatchOutcome::Skipped);
    assert_eq!(f.tables.indexed_blocks(fourth, 1), 0);

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
    let row = f.write(rank_facts(OTHER, OTHER, 0));
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
    let moved = f.write(rank_facts(OTHER, URL, 1));
    assert_eq!(moved, row);
    let store = Data::Stored(wide_store(1, 16));
    assert_eq!(f.admit(1, 2, store), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(row, 1), 1);

    assert!(f.host.open(ticket, &worker, MODEL).is_none());
    assert_eq!(f.tables.indexed_blocks(row, 1), 1);
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
    let moved = f.write(rank_facts(UPPER, OTHER, 1));
    assert_eq!(moved, upper);
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
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);

    // The removal ends: the row is cleared and forgotten, and the
    // other engine's next batch takes it.
    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
    assert_eq!(late(3, 3), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    let rows = f.sink.rows.lock();
    let stream = rows.written.get(&upper).map(|written| written.stream);
    assert_eq!(stream, Some(other));
    assert_eq!(rows.fed.get(&other), Some(&vec![upper]));
    assert_eq!(rows.fed.get(&f.lower), None);
}

/// A feed can open for a row its leaving stream still clears: the row
/// was a rank of that stream's engine and is laid again as rank 0 of its
/// own. The opening leaves the row to the removal, and the feed's batch
/// is passed over. Its first batch behind the removal's end begins the
/// row.
#[test]
fn a_feed_opened_under_a_leaving_stream_begins_its_row_behind_the_clear() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 5, store(1)), BatchOutcome::Applied);
    // The row is laid again in place as rank 0 of its own engine.
    assert_eq!(f.write(rank_facts(UPPER, UPPER, 0)), upper);
    // The first engine's stream is leaving: its removal marked the row
    // and has not cleared it yet. The row's own feed opens there.
    f.sink
        .rows
        .lock()
        .written
        .get_mut(&upper)
        .expect("the row's entry")
        .leaving = true;
    let worker = f
        .host
        .objects
        .registry()
        .get_by_url(UPPER)
        .expect("the row's object");
    let ticket = f.host.enroll(&worker, MODEL);
    let (sink, row) = f.host.open(ticket, &worker, MODEL).expect("the row's feed");
    assert_eq!(row, upper);
    // One batch on the row's own stream.
    let own = |sequence, hash| {
        let one = batch([store(hash)]);
        sink.admit(upper, 0, facts(sequence), &one)
    };
    assert_eq!(own(1, 2), BatchOutcome::Skipped);

    // The removal ends: the row is cleared and forgotten, and the
    // feed's next batch begins it.
    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(upper, 0), 0);
    assert_eq!(own(2, 3), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(upper, 0), 1);
    let rows = f.sink.rows.lock();
    let stream = rows.written.get(&upper).map(|written| written.stream);
    assert_eq!(stream, Some(upper));
    assert!(rows.fed.is_empty());
}

/// A row named under another address has two feeds open for a time: a
/// cut ends its object at the first address and lays one at the second,
/// whose feed opens while the first's removal is not told. That removal
/// leaves a feed open and takes nothing: the second feed's block stands,
/// and its next batch is taken. The second's own removal is the row's
/// last, and clears it.
#[test]
fn a_removal_that_leaves_a_feed_open_takes_nothing() {
    let f = Ranked::new();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    // A cut names the row's id under another address.
    let mut cut = f.store.checkpoint();
    let moved = cut.rows.iter_mut().find(|row| row.id == f.lower);
    moved.expect("the row").facts = rank_facts(OTHER, OTHER, 0);
    assert!(f.host.objects.watch_from(&f.store, cut));
    let registry = f.host.objects.registry();
    assert!(registry.get_by_url(URL).is_none(), "the first address");
    let second = registry.get_by_url(OTHER).expect("the second's object");
    let ticket = f.host.enroll(&second, MODEL);
    assert_eq!(ticket, Ok(f.lower));
    assert!(f.host.open(ticket, &second, MODEL).is_some());
    assert_eq!(f.admit(0, 1, store(1)), BatchOutcome::Applied);

    // The first address's removal, told behind the second feed's opening.
    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(f.lower, 0), 1);
    assert_eq!(f.admit(0, 2, store(2)), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(f.lower, 0), 2);

    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(f.lower, 0), 0);
    assert_eq!(f.admit(0, 3, store(3)), BatchOutcome::Skipped);
    let rows = f.sink.rows.lock();
    assert!(rows.written.is_empty() && rows.fed.is_empty());
}

/// A feed's opening releases the rows of other ranks that its row's
/// stream still writes and reaches no longer. A cut names the row under
/// another address, and rank 1's row stays at the first: the holder
/// routes it no batch of the row's stream. The second address's feed
/// opens while the first's removal is not told. Behind that opening rank
/// 1's row holds no block, the adapter keeps no entry of it, and the sum
/// the row's gauge is set from counts its own row alone. The first
/// address's removal then takes nothing, and the second's clears the row.
#[test]
fn a_feeds_opening_releases_the_rank_rows_its_stream_reaches_no_longer() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(0, 1, store(1)), BatchOutcome::Applied);
    assert_eq!(f.admit(1, 1, store(2)), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (1, 1));
    f.cut_to(OTHER);
    assert_eq!(f.sink.routed(f.lower, 1), None, "the second has no rank 1");
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1, "the cut leaves it");

    f.open_at(OTHER);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
    {
        let rows = f.sink.rows.lock();
        assert!(!rows.written.contains_key(&upper));
        assert!(rows.fed.get(&f.lower).is_none_or(Vec::is_empty));
    }
    // The gauge moves at a batch, to the sum of what the stream's rows
    // hold.
    assert_eq!(f.admit(0, 2, store(3)), BatchOutcome::Applied);
    assert_eq!(f.sink.held(f.lower), 1);

    // The first address's removal, told behind the second feed's opening.
    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(f.lower, 0), 1);
    assert_eq!(f.admit(0, 3, store(4)), BatchOutcome::Applied);
    f.sink.removed(f.lower);
    assert_eq!(f.blocks(upper), (0, 0));
}

/// Two feeds of one address keep their rank rows. A second feed opens
/// for the same object ahead of the first's removal: the holder names
/// rank 1's row among the rows of the row's engine, and the opening
/// releases none. Its block stands through the opening and through the
/// first feed's removal, and the stream's next batch of rank 1 is taken.
#[test]
fn two_feeds_of_one_address_keep_their_rank_rows() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    f.open_at(URL);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    assert_eq!(f.sink.rows.lock().fed.get(&f.lower), Some(&vec![upper]));

    f.sink.removed(f.lower);
    assert_eq!(f.admit(1, 2, store(2)), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 2);
    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
}

/// A released row laid again as a rank of the row's engine begins from
/// nothing: the row's stream takes it at its next batch of that rank, as
/// a row it writes for the first time, and lists it again.
#[test]
fn a_released_row_laid_again_as_a_rank_of_the_engine_begins_from_nothing() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    f.cut_to(OTHER);
    f.open_at(OTHER);
    assert!(!f.sink.rows.lock().written.contains_key(&upper));

    // The released row is laid again as rank 1 of the second address.
    assert_eq!(f.write(rank_facts(UPPER, OTHER, 1)), upper);
    assert_eq!(f.sink.routed(f.lower, 1), Some(upper));
    assert_eq!(f.admit(1, 2, store(2)), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    assert_eq!(f.sink.rows.lock().fed.get(&f.lower), Some(&vec![upper]));
}

/// An opening holds each row its stream still writes against the
/// holder's answer by itself. The second address has a rank 1 row of its
/// own, which the row's stream writes too: the opening keeps that row
/// and its block, and releases the first address's.
#[test]
fn an_opening_keeps_its_engines_rank_row_and_releases_the_other() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    f.cut_to(OTHER);
    let second = f.write(rank_facts(FOURTH, OTHER, 1));
    assert_eq!(f.sink.routed(f.lower, 1), Some(second));
    assert_eq!(f.admit(1, 2, store(2)), BatchOutcome::Applied);
    let both = vec![upper, second];
    assert_eq!(f.sink.rows.lock().fed.get(&f.lower), Some(&both));

    f.open_at(OTHER);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
    assert_eq!(f.tables.indexed_blocks(second, 1), 1);
    let rows = f.sink.rows.lock();
    assert!(!rows.written.contains_key(&upper));
    assert_eq!(rows.fed.get(&f.lower), Some(&vec![second]));
}

/// A feed's own row laid again in place as rank 0 of another engine
/// releases its rank rows at the new engine's feed's opening. The
/// holder's own write of the row's facts makes the state: no cut is built
/// by hand.
#[test]
fn a_row_laid_again_under_another_engine_releases_its_rank_rows() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    assert_eq!(f.write(rank_facts(URL, OTHER, 0)), f.lower);
    assert_eq!(f.sink.routed(f.lower, 1), None, "the engine has no rank 1");
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1, "the write leaves it");

    f.open_at(URL);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
    assert!(!f.sink.rows.lock().written.contains_key(&upper));
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
    assert!(f.sink.applied(f.lower, (upper, 1)).is_none());
    let own = batch([store(2)]);
    assert_eq!(
        f.sink.admit(upper, 0, facts(1), &own),
        BatchOutcome::Applied
    );
    assert_eq!(f.tables.indexed_blocks(upper, 0), 1);
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

/// What `seen` reads while `step` stands still inside its clear of
/// `row`, the row of `rank`. The tables' own lock is the probe: a clear
/// of a row takes its lane's lock, and here a batch's body holds that
/// lock until `seen` has answered, so the step cannot get past its clear
/// before then.
fn at_its_clear<T>(
    tables: &KvTables,
    row: WorkerId,
    rank: u16,
    sequence: u64,
    step: impl FnOnce() + Send,
    seen: impl Fn() -> Option<T>,
) -> T {
    let (entered, at_body) = mpsc::channel();
    let (release, held) = mpsc::channel::<()>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let outcome = tables.admit_through(row, rank, facts(sequence), &mut |_| {
                entered.send(()).expect("the case waits for the body");
                held.recv().expect("the case lets the body go");
            });
            assert_eq!(outcome, BatchOutcome::Applied);
        });
        at_body.recv().expect("the body holds the lane");
        let stepping = scope.spawn(step);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let read = loop {
            if let Some(read) = seen() {
                break Some(read);
            }
            if std::time::Instant::now() >= deadline {
                break None;
            }
            std::thread::yield_now();
        };
        // The body is let go before any failure is said, so a case that
        // fails ends: no thread of it waits on the lane.
        release.send(()).expect("the body waits");
        stepping.join().expect("the step ends");
        read.expect("the step never showed what the case waits for")
    })
}

/// A row's first clear runs in the hold that writes its entry, at a
/// stream's take and at a feed's open. No stream claims the row, and
/// writes it, between the two, so the clear takes nothing a later
/// stream wrote. A removal's clear runs outside the lock, behind its
/// mark: it takes every block the stream stored.
#[test]
fn a_claimed_rows_first_clear_runs_in_the_hold_that_writes_its_entry() {
    let f = Ranked::new();
    let row = f.upper();
    let (sink, tables, lower) = (&f.sink, &f.tables, f.lower);
    // One look at the adapter's list while the step cannot get past its
    // clear. Held: the step holds it, at its clear or still ahead of
    // it, so the case looks again behind a wait that is long against a
    // claim's own hold. A clear inside the hold still holds the list
    // then. Free, with the row's entry naming `stream`: the step wrote
    // its entry and let the list go ahead of its clear.
    let held_at_its_clear = |stream: WorkerId| {
        let Some(rows) = sink.rows.try_lock() else {
            std::thread::sleep(std::time::Duration::from_millis(20));
            return Some(sink.rows.try_lock().is_none());
        };
        let written = rows.written.get(&row);
        let wrote = written.is_some_and(|written| written.stream == stream);
        wrote.then_some(false)
    };

    // The engine's stream takes the row as its rank 1. The row is then
    // laid again as rank 0 of its own engine, and its own feed opens.
    let take = || assert!(sink.take(lower, (row, 1)).is_some());
    let held = at_its_clear(tables, row, 1, 1, take, || held_at_its_clear(lower));
    assert!(held, "a take's clear runs in the hold of its claim");
    assert_eq!(f.write(rank_facts(UPPER, UPPER, 0)), row);
    let begin = || assert_eq!(sink.begin(row, || true), Ok(()));
    let held = at_its_clear(tables, row, 0, 1, begin, || held_at_its_clear(row));
    assert!(held, "a begin's clear runs in the hold of its claim");

    // The feed's removal: its mark stands in the list, which is free,
    // while its clear still waits for the lane.
    let marked = || {
        let rows = sink.rows.try_lock()?;
        let leaving = rows
            .written
            .get(&row)
            .is_some_and(|written| written.leaving);
        leaving.then_some(())
    };
    at_its_clear(tables, row, 0, 1, || sink.removed(row), marked);
}

/// A rank's row that an opening releases bears its mark while its clear
/// runs: the list is free then, and no stream takes the row until the
/// clear has ended and the row is forgotten.
#[test]
fn a_released_row_bears_its_mark_while_its_clear_runs() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    f.cut_to(OTHER);
    let (sink, lower) = (&f.sink, f.lower);
    let marked = || {
        let rows = sink.rows.try_lock()?;
        let leaving = rows
            .written
            .get(&upper)
            .is_some_and(|written| written.leaving);
        leaving.then_some(())
    };
    // The second address's feed begins the row, and its release's clear
    // of rank 1's row waits for that row's lane.
    let begin = || assert_eq!(sink.begin(lower, || true), Ok(()));
    at_its_clear(&f.tables, upper, 1, 2, begin, marked);
    assert!(!sink.rows.lock().written.contains_key(&upper));
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
}

/// A batch that took a row's state ahead of the row's release stores
/// nothing behind it. A batch of rank 1 reads its feed's record and its
/// row's state, and the cut and the second address's opening then
/// release the row. The batch goes on behind that: it is passed over,
/// the row holds no block, the adapter keeps no entry of it, and the
/// sum the stream's gauge is set from stands as it was.
#[test]
fn a_batch_that_took_a_rows_state_ahead_of_its_release_stores_nothing() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(0, 1, store(1)), BatchOutcome::Applied);
    assert_eq!(f.admit(1, 1, store(2)), BatchOutcome::Applied);
    let feed = f.feed();
    let state = f.sink.applied(f.lower, (upper, 1));
    let state = state.expect("the row's state");
    f.cut_to(OTHER);
    f.open_at(OTHER);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
    let held = f.sink.held(f.lower);

    let (late, place) = (batch([store(3)]), (upper, 1));
    let outcome = f
        .sink
        .admit_under(f.lower, place, &feed, state, facts(2), &late);
    assert_eq!(outcome, BatchOutcome::Skipped);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
    assert!(!f.sink.rows.lock().written.contains_key(&upper));
    assert_eq!(f.sink.held(f.lower), held);
}

/// The same at a stream's removal, for its own row. A batch of rank 0
/// reads its feed's record and its row's state, which the adapter keeps,
/// and the removal is told. The batch goes on behind it: it is passed
/// over, the row holds no block and the adapter keeps no entry of it.
#[test]
fn a_batch_that_took_its_rows_state_ahead_of_its_removal_stores_nothing() {
    let f = Ranked::new();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(0, 1, store(1)), BatchOutcome::Applied);
    let feed = f.feed();
    let state = f.sink.applied(f.lower, (f.lower, 0));
    let state = state.expect("the row's state");
    assert!(state.1, "the adapter keeps the state");
    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(f.lower, 0), 0);

    let (late, place) = (batch([store(2)]), (f.lower, 0));
    let outcome = f
        .sink
        .admit_under(f.lower, place, &feed, state, facts(2), &late);
    assert_eq!(outcome, BatchOutcome::Skipped);
    assert_eq!(f.tables.indexed_blocks(f.lower, 0), 0);
    assert!(!f.sink.rows.lock().written.contains_key(&f.lower));
    assert_eq!(f.sink.held(f.lower), 0);
}

/// A leaving row's mark bears a released state. A batch of the stream
/// that asks for the row's state while the row's release stands in its
/// clear gets the mark's, and stores nothing under it behind the
/// release's end.
#[test]
fn a_batch_that_reads_a_leaving_rows_mark_stores_nothing() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    let feed = f.feed();
    f.cut_to(OTHER);
    let (sink, lower) = (&f.sink, f.lower);
    // What a batch of the stream gets for the row once its mark stands.
    let marked = || {
        let leaving = sink.rows.try_lock()?.written.get(&upper)?.leaving;
        leaving.then(|| sink.applied(lower, (upper, 1)))
    };
    let begin = || assert_eq!(sink.begin(lower, || true), Ok(()));
    let state = at_its_clear(&f.tables, upper, 1, 2, begin, marked);
    let state = state.expect("the stream's batch gets the mark's state");
    assert!(state.0.lock().released, "the mark's state is released");

    let late = batch([store(2)]);
    let outcome = sink.admit_under(lower, (upper, 1), &feed, state, facts(3), &late);
    assert_eq!(outcome, BatchOutcome::Skipped);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
    assert!(!sink.rows.lock().written.contains_key(&upper));
}

/// A leaving row's state is released ahead of the row's clear. A
/// batch's body holds the row's lane, so the release's clear waits, and
/// the state the stream held of the row reads released by then.
#[test]
fn a_leaving_rows_state_is_released_ahead_of_its_clear() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    let state = f.sink.applied(f.lower, (upper, 1));
    let (state, _) = state.expect("the row's state");
    f.cut_to(OTHER);
    let (sink, lower) = (&f.sink, f.lower);
    let released = || state.try_lock()?.released.then_some(());
    let begin = || assert_eq!(sink.begin(lower, || true), Ok(()));
    at_its_clear(&f.tables, upper, 1, 2, begin, released);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
}

/// A batch that holds a released state is passed over ahead of the row's
/// cursor. The released row is laid again as a rank of the row's engine,
/// and the stream writes it anew at sequence 10. A late batch at 1,
/// which the cursor would read as a restart, leaves the row's block and
/// the cursor as they were. So does one at 12, which asks no replay, and
/// a late chunk of a snapshot. The stream's batch at 11 is taken.
#[test]
fn a_batch_that_holds_a_released_state_leaves_the_cursor_where_it_was() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    let feed = f.feed();
    let state = f.sink.applied(f.lower, (upper, 1));
    let (state, kept) = state.expect("the row's state");
    f.cut_to(OTHER);
    f.open_at(OTHER);
    assert_eq!(f.write(rank_facts(UPPER, OTHER, 1)), upper);
    assert_eq!(f.admit(1, 10, store(2)), BatchOutcome::Applied);

    let (late, place) = (batch([store(3)]), (upper, 1));
    let applied = || f.sink.cursor(f.lower, 1).map(|cursor| cursor.applied);
    let chunk = BatchFacts {
        snapshot: Some(SnapshotChunk::default()),
        ..facts(20)
    };
    for header in [facts(1), facts(12), chunk] {
        let held = (Arc::clone(&state), kept);
        let outcome = f
            .sink
            .admit_under(f.lower, place, &feed, held, header, &late);
        assert_eq!(outcome, BatchOutcome::Skipped);
        assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
        assert_eq!(applied(), Some(10));
    }
    assert_eq!(f.admit(1, 11, store(4)), BatchOutcome::Applied);
}

/// Two batches of one stream that both missed a row's entry take one
/// state. Two feeds of one row can both ask for a rank's row no entry is
/// kept of. The second take hands the state the first made and clears
/// nothing, so the first batch's store stands.
#[test]
fn two_batches_that_both_missed_a_rows_entry_take_one_state() {
    let f = Ranked::new();
    let upper = f.upper();
    let feed = f.feed();
    let place = (upper, 1);
    let first = f.sink.take(f.lower, place).expect("the stream takes it");
    let stored = batch([Data::Stored(wide_store(1, 16))]);
    let held = (Arc::clone(&first), true);
    let outcome = f
        .sink
        .admit_under(f.lower, place, &feed, held, facts(1), &stored);
    assert_eq!(outcome, BatchOutcome::Applied);
    let second = f.sink.take(f.lower, place).expect("the stream's row");
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1, "no second clear");
}

/// A holder's stand-in that stands a row at rank 0 and then names no row
/// of its engine: what an opening could read between the two holds of a
/// row's object laid again, were it to ask twice. It counts how often
/// an engine's rows are asked.
struct Between {
    upper: WorkerId,
    asked: AtomicU32,
}

impl Ranks for Between {
    fn row_of(&self, row: WorkerId, rank: u16) -> Option<WorkerId> {
        Some(if rank == 0 { row } else { self.upper })
    }

    fn rows_of(&self, _: WorkerId) -> Vec<WorkerId> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        Vec::new()
    }
}

/// An opening asks the holder's ranks once: the rows of its row's engine.
/// Where the holder names none, the row stands at rank 0 no longer. The
/// opening is refused, and it releases no row its stream writes.
#[test]
fn an_opening_the_holder_names_no_rows_for_is_refused_and_releases_nothing() {
    let store = Arc::new(MemoryStore::new());
    let tables = Arc::new(KvTables::default());
    let lower = store.upsert(rank_facts(URL, URL, 0)).id();
    let upper = store.upsert(rank_facts(UPPER, URL, 1)).id();
    for id in [lower, upper] {
        tables.member(&store.get(id).expect("the row"));
    }
    let asked = AtomicU32::new(0);
    let between = Arc::new(Between { upper, asked });
    let ranks: Arc<dyn Ranks> = Arc::clone(&between) as _;
    let sink = KvAdapter::over(Arc::clone(&tables), Some(ranks));
    // The stream's record stands, and it writes rank 1's row.
    sink.opened(lower, MODEL, URL);
    let stored = batch([Data::Stored(wide_store(1, 16))]);
    let outcome = sink.admit(lower, 1, facts(1), &stored);
    assert_eq!(outcome, BatchOutcome::Applied);

    assert_eq!(sink.begin(lower, || true), Err(SkipReason::UpperRank));
    assert_eq!(between.asked.load(Ordering::Relaxed), 1, "one ask");
    assert_eq!(tables.indexed_blocks(upper, 1), 1);
    assert_eq!(sink.rows.lock().fed.get(&lower), Some(&vec![upper]));
}

/// A route read ahead of a cut takes no released row back. The cut and
/// the second address's opening release rank 1's row, and the release
/// ends. A batch of the row's stream that read its route to that row
/// ahead of the cut asks for the row's state behind that. The holder
/// routes it there no longer: the batch gets no state, and the adapter
/// keeps no entry of the row.
#[test]
fn a_route_read_ahead_of_a_cut_takes_no_released_row_back() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = Data::Stored(wide_store(1, 16));
    assert_eq!(f.admit(1, 1, store), BatchOutcome::Applied);
    f.cut_to(OTHER);
    f.open_at(OTHER);
    assert!(f.sink.applied(f.lower, (upper, 1)).is_none());
    let rows = f.sink.rows.lock();
    assert!(!rows.written.contains_key(&upper));
    assert!(rows.fed.get(&f.lower).is_none_or(Vec::is_empty));
}

/// A batch of a row's old stream that took the row's state ahead of the
/// row's move stores nothing behind it. Rank 1's row moves under another
/// engine at the same rank, so its lane stands. That engine's stream
/// takes the row and writes it anew at sequence 10. The old stream's
/// late batch at sequence 1, which the cursor would read as a restart,
/// leaves the new stream's block and the cursor as they were.
#[test]
fn a_batch_that_took_a_rows_state_ahead_of_its_move_stores_nothing() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 5, store(1)), BatchOutcome::Applied);
    let feed = f.feed();
    let held = f.sink.applied(f.lower, (upper, 1));
    let held = held.expect("the row's state");
    let other = f.other(OTHER);
    assert_eq!(f.write(rank_facts(UPPER, OTHER, 1)), upper);
    let anew = f.sink.admit(other, 1, facts(10), &batch([store(2)]));
    assert_eq!(anew, BatchOutcome::Applied);

    let (late, place) = (batch([store(3)]), (upper, 1));
    let outcome = f
        .sink
        .admit_under(f.lower, place, &feed, held, facts(1), &late);
    assert_eq!(outcome, BatchOutcome::Skipped);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    let applied = f.sink.cursor(other, 1).map(|cursor| cursor.applied);
    assert_eq!(applied, Some(10));
}

/// A saved route takes no row the holder names another for. Rank 1's
/// row moves under another engine, whose stream writes it, and a second
/// row is laid as the first engine's rank 1. A batch of the first
/// stream that read its route to the first row ahead of that asks for
/// the row's state. It gets none: the row stays the other stream's, with
/// its block.
#[test]
fn a_saved_route_takes_no_row_the_holder_names_another_for() {
    let f = Ranked::new();
    let upper = f.upper();
    let other = f.other(OTHER);
    assert_eq!(f.write(rank_facts(UPPER, OTHER, 1)), upper);
    let stored = batch([Data::Stored(wide_store(1, 16))]);
    let anew = f.sink.admit(other, 1, facts(1), &stored);
    assert_eq!(anew, BatchOutcome::Applied);
    let second = f.write(rank_facts(THIRD, URL, 1));
    assert_eq!(f.sink.routed(f.lower, 1), Some(second));

    assert!(f.sink.applied(f.lower, (upper, 1)).is_none());
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    let rows = f.sink.rows.lock();
    let stream = rows.written.get(&upper).map(|written| written.stream);
    assert_eq!(stream, Some(other));
    assert!(rows.fed.get(&f.lower).is_none_or(Vec::is_empty));
}

/// A moved row's state is released ahead of the row's clear, as a
/// leaving row's is. Rank 1's row moves under another engine at the
/// same rank. A batch's body holds the row's lane, so the clear of that
/// engine's take waits, and the state the old stream held of the row
/// reads released by then.
#[test]
fn a_moved_rows_state_is_released_ahead_of_its_clear() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = Data::Stored(wide_store(1, 16));
    assert_eq!(f.admit(1, 1, store), BatchOutcome::Applied);
    let state = f.sink.applied(f.lower, (upper, 1));
    let (state, _) = state.expect("the row's state");
    let other = f.other(OTHER);
    assert_eq!(f.write(rank_facts(UPPER, OTHER, 1)), upper);
    let sink = &f.sink;
    let released = || state.try_lock()?.released.then_some(());
    let take = || assert!(sink.take(other, (upper, 1)).is_some());
    at_its_clear(&f.tables, upper, 1, 2, take, released);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
}

/// A feed's opening that takes a row another stream wrote releases the
/// state that stream wrote it under, as a take does. Rank 1's row is
/// laid again as rank 0 of its own engine, and its feed opens. The row
/// is then laid back as rank 1 of the first engine, so a lane of that
/// rank stands for it again. A batch of the first stream that took the
/// row's state ahead of the opening goes on behind the lay: it is
/// passed over, and the row holds no block.
#[test]
fn a_batch_that_took_a_rows_state_ahead_of_its_feeds_opening_stores_nothing() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    let feed = f.feed();
    let held = f.sink.applied(f.lower, (upper, 1));
    let held = held.expect("the row's state");
    assert_eq!(f.other(UPPER), upper);
    assert_eq!(f.upper(), upper);
    assert_eq!(f.sink.routed(f.lower, 1), Some(upper));

    let (late, place) = (batch([store(2)]), (upper, 1));
    let outcome = f
        .sink
        .admit_under(f.lower, place, &feed, held, facts(2), &late);
    assert_eq!(outcome, BatchOutcome::Skipped);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
}

/// A holder's stand-in that routes every rank it is asked for to one
/// row, and counts its asks by whether the adapter's rows were locked
/// then.
struct Asked {
    upper: WorkerId,
    /// The adapter whose lock is tried, set once it is built.
    sink: Mutex<std::sync::Weak<KvAdapter>>,
    /// The asks made while the rows' lock was free.
    free: AtomicU32,
    /// The asks made inside a hold of the rows' lock.
    held: AtomicU32,
}

impl Ranks for Asked {
    fn row_of(&self, _: WorkerId, _: u16) -> Option<WorkerId> {
        let sink = self.sink.lock().upgrade().expect("the adapter");
        let held = sink.rows.try_lock().is_none();
        let count = if held { &self.held } else { &self.free };
        count.fetch_add(1, Ordering::Relaxed);
        Some(self.upper)
    }

    fn rows_of(&self, _: WorkerId) -> Vec<WorkerId> {
        Vec::new()
    }
}

/// A take asks the holder for its route in the hold that writes the
/// row's entry: the rows are locked while that answer is read. A batch
/// reads its route first ahead of that lock, and a batch on a row its
/// stream writes asks that once and no more.
#[test]
fn a_take_asks_the_holder_in_the_hold_that_writes_its_row() {
    let store = Arc::new(MemoryStore::new());
    let tables = Arc::new(KvTables::default());
    let lower = store.upsert(rank_facts(URL, URL, 0)).id();
    let upper = store.upsert(rank_facts(UPPER, URL, 1)).id();
    for id in [lower, upper] {
        tables.member(&store.get(id).expect("the row"));
    }
    let asked = Arc::new(Asked {
        upper,
        sink: Mutex::default(),
        free: AtomicU32::new(0),
        held: AtomicU32::new(0),
    });
    let ranks: Arc<dyn Ranks> = Arc::clone(&asked) as _;
    let sink = Arc::new(KvAdapter::over(Arc::clone(&tables), Some(ranks)));
    *asked.sink.lock() = Arc::downgrade(&sink);
    let asks = || {
        let free = asked.free.load(Ordering::Relaxed);
        (free, asked.held.load(Ordering::Relaxed))
    };
    sink.opened(lower, MODEL, URL);
    let stored = |hash| batch([Data::Stored(wide_store(hash, 16))]);

    // The stream's first batch of rank 1 takes the row.
    let outcome = sink.admit(lower, 1, facts(1), &stored(1));
    assert_eq!(outcome, BatchOutcome::Applied);
    assert_eq!(asks(), (1, 1), "its route, then the take's read");
    // The stream writes the row now.
    let outcome = sink.admit(lower, 1, facts(2), &stored(2));
    assert_eq!(outcome, BatchOutcome::Applied);
    assert_eq!(asks(), (2, 1), "its route alone");
}
