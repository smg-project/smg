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
/// `row`. The tables' own lock is the probe: a clear of a row takes its
/// lane's lock, and here a batch's body holds that lock until `seen`
/// has answered, so the step cannot get past its clear before then.
fn at_its_clear<T>(
    tables: &KvTables,
    row: WorkerId,
    sequence: u64,
    step: impl FnOnce() + Send,
    seen: impl Fn() -> Option<T>,
) -> T {
    let (entered, at_body) = mpsc::channel();
    let (release, held) = mpsc::channel::<()>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let outcome = tables.admit_through(row, 0, facts(sequence), &mut |_| {
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
    let store = Arc::new(MemoryStore::new());
    let tables = Arc::new(KvTables::default());
    let sink = Arc::new(KvAdapter::new(Arc::clone(&tables)));
    let row = store.upsert(rank_facts(URL, URL, 0)).id();
    let other = store.upsert(rank_facts(OTHER, OTHER, 0)).id();
    for id in [row, other] {
        tables.member(&store.get(id).expect("the row"));
    }
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

    // Another stream's take of the row, then the open of its own feed.
    let take = || assert!(sink.take(other, row).is_some());
    let held = at_its_clear(&tables, row, 1, take, || held_at_its_clear(other));
    assert!(held, "a take's clear runs in the hold of its claim");
    let begin = || assert_eq!(sink.begin(row, || true), Ok(()));
    let held = at_its_clear(&tables, row, 1, begin, || held_at_its_clear(row));
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
    at_its_clear(&tables, row, 1, || sink.removed(row), marked);
}
