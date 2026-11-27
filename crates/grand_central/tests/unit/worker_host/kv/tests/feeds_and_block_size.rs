//! A host's feeds: the row a ticket opens a feed for, what it begins from,
//! and a batch on a stream with no feed open. The block size: the one a
//! model's batches name, until the model leaves.

use super::*;

/// A batch the cursor takes names the model's block size, whatever the
/// worker stated at its registration: none, a narrower size or a wider
/// one. A narrower store behind it moves nothing, and a batch the cursor
/// skips names none.
#[test]
fn a_batch_the_store_takes_names_the_models_block_size() {
    for stated in [None, Some(16), Some(64)] {
        let f = Fixture::single(stated);
        let (_host, sink) = opened(&f);
        assert_eq!(f.tables.block_size(MODEL), stated);
        let mut first = wide(1, 16);
        first.events.extend(wide(2, 32).events);
        assert_eq!(
            sink.admit(f.row, 0, facts(1), &first),
            BatchOutcome::Applied
        );
        assert_eq!(f.tables.block_size(MODEL), Some(32), "stated: {stated:?}");
        // A narrower store is a prefix entry.
        assert_eq!(
            sink.admit(f.row, 0, facts(2), &wide(3, 8)),
            BatchOutcome::Applied
        );
        assert_eq!(f.tables.block_size(MODEL), Some(32));
        // A duplicate of the batch the row holds is skipped.
        assert_eq!(
            sink.admit(f.row, 0, facts(2), &wide(4, 64)),
            BatchOutcome::Skipped
        );
        assert_eq!(f.tables.block_size(MODEL), Some(32));
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
    assert_eq!(f.tables.block_size(MODEL), Some(16));
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
        f.tables.block_size(MODEL)
    };
    assert_eq!(stored(16), Some(16));
    // Another worker of the model registers and states a wider size.
    f.tables.set_block_size(MODEL, 64);
    assert_eq!(f.tables.block_size(MODEL), Some(64));
    assert_eq!(stored(16), Some(16));
    // The model leaves, and comes back with a registration's size.
    host.model_left(MODEL);
    assert_eq!(f.tables.block_size(MODEL), None);
    f.tables.set_block_size(MODEL, 64);
    assert_eq!(stored(16), Some(16));
    // A wider block raises it.
    assert_eq!(stored(32), Some(32));
    // A prefix entry behind a wider registration puts the engine's block
    // back, not its own width.
    f.tables.set_block_size(MODEL, 64);
    assert_eq!(stored(8), Some(32));
}

#[test]
fn the_block_size_is_read_and_set_at_the_store() {
    let f = Fixture::new();
    f.tables.set_block_size(MODEL, 16);
    assert_eq!(f.sink.block_size(MODEL), Some(16));
    f.sink.set_block_size(MODEL, 32);
    assert_eq!(f.tables.block_size(MODEL), Some(32));
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
    assert_eq!(f.tables.block_size(MODEL), Some(BLOCK_SIZE));
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
        f.tables.cursor(f.row, 0).map(|cursor| cursor.applied),
        Some(7)
    );
    assert_eq!(f.tables.indexed_blocks(f.row, 0), 1);

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
    assert_eq!(f.tables.indexed_blocks(row, 0), 0);
    assert!(begun(&sink, row));

    assert_eq!(sink.admit(row, 0, facts(9), &stores), BatchOutcome::Applied);
    let (sink, row) = host
        .open(host.enroll(&worker, MODEL), &worker, MODEL)
        .expect("the row's feed");
    assert_eq!(sink.cursor(row, 0).map(|cursor| cursor.applied), Some(9));
    assert_eq!(f.tables.indexed_blocks(row, 0), 1);
    assert_eq!(
        sink.admit(
            row,
            0,
            facts(10),
            &batch([Data::Removed(group_removal(&[1], 2))])
        ),
        BatchOutcome::Applied
    );
    assert_eq!(f.tables.indexed_blocks(row, 0), 1);
}

/// The tables keep a row's lane at the row's own rank. A row the holder
/// stands above rank 0 is its engine's rank 0 stream's. A ticket for
/// it opens no feed, and what the tables keep of the row stands. It is
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
        f.tables.cursor(f.row, RANK).map(|cursor| cursor.applied),
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
        f.tables.cursor(f.row, RANK).map(|cursor| cursor.applied),
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
    assert_eq!(f.tables.block_size(MODEL), Some(BLOCK_SIZE));
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
    let moved = WorkerFacts {
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
    };
    let written = host.objects.upsert(&f.store, moved, Registrant::Control);
    assert!(written.is_some(), "the store takes the write");
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

/// A host over a holder that was handed no tables indexes nothing. The
/// row's feed opens all the same, since a stream brings more than
/// blocks. No row holds a lane, so its batch is passed over and no
/// cursor stands for it.
#[test]
fn a_host_over_a_holder_handed_no_tables_indexes_nothing() {
    let f = Fixture::single(Some(BLOCK_SIZE));
    let objects = Arc::new(Objects::new(Arc::new(WorkerRegistry::new())));
    lay_standing(&objects, &f.store);
    assert!(objects.kv_tables().is_none());
    let host: KvHost<Members> = KvHost::new(objects);
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let (sink, row) = host
        .open(host.enroll(&worker, MODEL), &worker, MODEL)
        .expect("the row's feed");
    assert_eq!(row, f.row);
    let one = batch([Data::Stored(stored(1))]);
    assert_eq!(sink.admit(row, 0, facts(1), &one), BatchOutcome::Skipped);
    assert_eq!(sink.cursor(row, 0), None);
    assert_eq!(sink.tables.indexed_blocks(row, 0), 0);
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
        assert_eq!(f.tables.block_size(MODEL), None);
    }
}

#[test]
fn a_model_that_left_forgets_its_block_size() {
    let f = Fixture::new();
    let host = f.host();
    f.tables.set_block_size(MODEL, 32);
    host.model_left(MODEL);
    assert_eq!(f.tables.block_size(MODEL), None);
    f.tables.set_block_size(MODEL, 16);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
}

/// A model that left takes its own word alone: another model keeps the
/// engine's word and the tables' size.
#[test]
fn a_model_that_left_takes_its_own_block_size_alone() {
    const OTHER_MODEL: &str = "n";
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let first = sink.admit(f.row, 0, facts(1), &wide(1, 32));
    assert_eq!(first, BatchOutcome::Applied);
    // What a feed of the other model learned.
    sink.sizes.lock().insert(Arc::from(OTHER_MODEL), 64);
    f.tables.set_block_size(OTHER_MODEL, 64);
    host.model_left(MODEL);
    assert_eq!(f.tables.block_size(MODEL), None);
    assert_eq!(f.tables.block_size(OTHER_MODEL), Some(64));
    let words = sink.sizes.lock();
    assert_eq!(words.get(MODEL), None);
    assert_eq!(words.get(OTHER_MODEL), Some(&64));
}

/// A model that left takes the engine's word with it: the first stored
/// block behind its return names its size, a narrower one too.
#[test]
fn a_model_that_left_names_its_block_size_anew() {
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let first = sink.admit(f.row, 0, facts(1), &wide(1, 32));
    assert_eq!(first, BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(32));
    // The model's one worker leaves, and the model with it.
    sink.removed(f.row);
    host.model_left(MODEL);
    assert_eq!(f.tables.block_size(MODEL), None);
    // It returns on the same host with a narrower block.
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let ticket = host.enroll(&worker, MODEL);
    let (sink, _) = host.open(ticket, &worker, MODEL).expect("the row's feed");
    let again = sink.admit(f.row, 0, facts(1), &wide(2, 16));
    assert_eq!(again, BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
}

/// A stream no feed was opened for writes nothing: its batch is passed
/// over, and the adapter keeps no row for it.
#[test]
fn a_batch_on_a_stream_no_feed_was_opened_for_is_passed_over() {
    let f = Fixture::single(None);
    // Another adapter over the same tables, which opened no feed.
    let unopened = KvAdapter::new(Arc::clone(&f.tables));
    assert_eq!(
        unopened.admit(f.row, 0, facts(1), &wide(1, 32)),
        BatchOutcome::Skipped
    );
    assert_eq!(f.tables.indexed_blocks(f.row, 0), 0);
    assert_eq!(f.tables.cursor(f.row, 0), None);
    assert!(unopened.rows.lock().written.is_empty());
}

/// A stream that left takes its feed's record with it, and a batch said
/// behind its removal is passed over. It writes nothing: no block, no
/// cursor and no entry of the adapter, and it names no size. The
/// worker's next feed opens on the same row and begins it again: its
/// first batch is taken below the cursor the row held.
#[test]
fn a_batch_behind_its_streams_removal_is_passed_over() {
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    assert_eq!(
        sink.admit(f.row, 0, facts(7), &wide(1, 16)),
        BatchOutcome::Applied
    );
    sink.removed(f.row);
    assert_eq!(
        sink.admit(f.row, 0, facts(8), &wide(2, 32)),
        BatchOutcome::Skipped
    );
    assert_eq!(f.tables.indexed_blocks(f.row, 0), 0);
    assert_eq!(sink.cursor(f.row, 0), None);
    assert_eq!(f.tables.block_size(MODEL), Some(16), "the wider late store");
    {
        let rows = sink.rows.lock();
        assert!(rows.written.is_empty() && rows.fed.is_empty());
    }

    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let ticket = host.enroll(&worker, MODEL);
    let (sink, row) = host.open(ticket, &worker, MODEL).expect("the row's feed");
    assert!(begun(&sink, row));
    assert_eq!(
        sink.admit(row, 0, facts(1), &wide(3, 16)),
        BatchOutcome::Applied
    );
    assert_eq!(f.tables.indexed_blocks(row, 0), 1);
}

/// A batch reads its feed's record where it begins and learns at its
/// end. One that began ahead of its stream's removal and learns behind it
/// names no size: the feed is its row's no longer, and the model's
/// departure cleared the size in between. A feed opened anew for the row
/// is another feed, so the old batch's learning is not its either, and
/// its own learning names the size.
#[test]
fn a_learning_behind_its_streams_removal_names_no_block_size() {
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let of_row = |sink: &KvAdapter| sink.feeds.lock().get(&f.row).cloned();
    let began = of_row(&sink).expect("the feed the batch found open");
    sink.removed(f.row);
    host.model_left(MODEL);
    sink.learn(f.row, &began, 32);
    assert_eq!(f.tables.block_size(MODEL), None);
    assert!(sink.sizes.lock().is_empty());
    // The model returns on the same host, under the same row.
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let ticket = host.enroll(&worker, MODEL);
    let (sink, _) = host.open(ticket, &worker, MODEL).expect("the row's feed");
    sink.learn(f.row, &began, 32);
    assert_eq!(f.tables.block_size(MODEL), None);
    let anew = of_row(&sink).expect("the feed opened anew");
    sink.learn(f.row, &anew, 16);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
}

/// A model's departure takes the engine's word and the tables' size in
/// one hold of the lock a learning holds, so no row of the model learns
/// between the two. The table's own lock is the probe: the case holds
/// it, and the departure cannot get past its call of the tables.
#[test]
fn a_models_departure_holds_the_learnings_lock_through_both_removals() {
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let first = sink.admit(f.row, 0, facts(1), &wide(1, 32));
    assert_eq!(first, BatchOutcome::Applied);
    // The engine's words, looked at while the departure stands at the
    // tables. Free, with the model's word gone: the departure let the
    // lock go ahead of the tables. Free, with the word still there: the
    // departure has not begun. Held at every look over a quarter second,
    // which is long against a removal's own hold: the departure holds
    // them, at the tables or still ahead of them.
    let held_at_the_tables = || {
        let window = std::time::Instant::now() + std::time::Duration::from_millis(250);
        loop {
            if let Some(sizes) = sink.sizes.try_lock() {
                return (!sizes.contains_key(MODEL)).then_some(false);
            }
            if std::time::Instant::now() >= window {
                return Some(true);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    };
    let held = std::thread::scope(|scope| {
        f.tables.locked(|| {
            // The collector's own call: the host's.
            scope.spawn(|| host.model_left(MODEL));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                if let Some(held) = held_at_the_tables() {
                    break Some(held);
                }
                if std::time::Instant::now() >= deadline {
                    break None;
                }
                std::thread::yield_now();
            }
        })
    });
    assert_eq!(held, Some(true), "let go ahead of the tables");
    assert_eq!(f.tables.block_size(MODEL), None);
    assert!(sink.sizes.lock().is_empty());
}
