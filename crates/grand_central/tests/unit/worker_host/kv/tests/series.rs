//! The cache-event series of the fixture's worker, read from a recorder
//! of the case's own. The recorder's crates come with the controller.

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle, PrometheusRecorder};

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

/// A batch the cursor takes counts its blocks by operation and
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

/// A recorder that asks two things at each counter it is asked for:
/// whether the adapter's records are free, and whether its row still
/// holds a block.
struct AsksAtItsCounter<'a> {
    sink: &'a KvAdapter,
    row: WorkerId,
    asked: AtomicU32,
    held: AtomicU32,
    ahead: AtomicU32,
}

impl metrics::Recorder for AsksAtItsCounter<'_> {
    fn describe_counter(
        &self,
        _: metrics::KeyName,
        _: Option<metrics::Unit>,
        _: metrics::SharedString,
    ) {
    }

    fn describe_gauge(
        &self,
        _: metrics::KeyName,
        _: Option<metrics::Unit>,
        _: metrics::SharedString,
    ) {
    }

    fn describe_histogram(
        &self,
        _: metrics::KeyName,
        _: Option<metrics::Unit>,
        _: metrics::SharedString,
    ) {
    }

    fn register_counter(&self, _: &metrics::Key, _: &metrics::Metadata<'_>) -> metrics::Counter {
        self.asked.fetch_add(1, Ordering::Relaxed);
        if self.sink.feeds.try_lock().is_none() {
            self.held.fetch_add(1, Ordering::Relaxed);
        }
        if self.sink.tables.indexed_blocks(self.row, 0) != 0 {
            self.ahead.fetch_add(1, Ordering::Relaxed);
        }
        metrics::Counter::noop()
    }

    fn register_gauge(&self, _: &metrics::Key, _: &metrics::Metadata<'_>) -> metrics::Gauge {
        metrics::Gauge::noop()
    }

    fn register_histogram(
        &self,
        _: &metrics::Key,
        _: &metrics::Metadata<'_>,
    ) -> metrics::Histogram {
        metrics::Histogram::noop()
    }
}

/// A reset writes its series last: behind the reset, with the adapter's
/// records let go. A recorder is the host's. A removal asks for the
/// records under the rows' lock, so a recorder that called the adapter
/// under the records could stop that removal. And a feed could open and
/// store inside the call of a recorder that ran ahead of the reset,
/// which then cleared that store. A row with no record writes no
/// series, whatever record another row has.
#[test]
fn a_reset_writes_its_series_last_with_the_records_let_go() {
    let f = Fixture::single(Some(BLOCK_SIZE));
    let (_host, sink) = opened(&f);
    assert_eq!(
        sink.admit(f.row, 0, facts(1), &wide(1, 32)),
        BatchOutcome::Applied
    );
    // A second row, with no feed open.
    let other = f.store.upsert(WorkerFacts {
        key: "http://b:8000".into(),
        engine: "http://b:8000".into(),
        incarnation: 1,
        models: vec![MODEL.into()],
        dp_size: 1,
        ..WorkerFacts::default()
    });
    let recorder = AsksAtItsCounter {
        sink: &sink,
        row: f.row,
        asked: AtomicU32::new(0),
        held: AtomicU32::new(0),
        ahead: AtomicU32::new(0),
    };
    let read = |count: &AtomicU32| count.load(Ordering::Relaxed);
    metrics::with_local_recorder(&recorder, || {
        sink.resync(other.id(), None, ResyncReason::DataLoss);
        assert_eq!(read(&recorder.asked), 0, "no record, no series");
        sink.resync(f.row, None, ResyncReason::DataLoss);
    });
    assert_eq!(read(&recorder.asked), 1, "one series");
    assert_eq!(read(&recorder.held), 0, "the records are free");
    assert_eq!(read(&recorder.ahead), 0, "the row is reset");
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
        assert_eq!(first_block_holders(&f, 1), [(f.row, 1)]);
    });
}

/// The rows that hold the block `hash` as the first block of a
/// request, each with its count of the request's leading blocks.
fn first_block_holders(f: &Fixture, hash: i64) -> Vec<(WorkerId, u32)> {
    let request = [kv_index::compute_content_hash(&tokens_of(hash)).0];
    f.tables.overlaps(MODEL, &request)
}

/// A worker that holds blocks stores under a parent it does not
/// hold: the store is counted for the worker and starts a chain of
/// its own, which a request reads from its first block.
#[test]
fn a_parentless_store_behind_held_blocks_is_counted_and_starts_a_chain() {
    let series = Series::new();
    metrics::with_local_recorder(&series.recorder, || {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let (_host, sink) = opened(&f);
        let count = |name: &str| series.read(name, &[]);
        assert_eq!(
            sink.admit(f.row, 0, facts(1), &wide(1, 32)),
            BatchOutcome::Applied
        );
        assert_eq!(count("smg_kv_event_parentless_stores_total"), None);
        let orphan = KvBlocksStored {
            parent_block_hash: Some(99),
            ..wide_store(2, 32)
        };
        let orphan = batch([Data::Stored(orphan)]);
        assert_eq!(
            sink.admit(f.row, 0, facts(2), &orphan),
            BatchOutcome::Applied
        );
        assert_eq!(count("smg_kv_event_parentless_stores_total"), Some(1.0));
        assert_eq!(count("smg_kv_event_parentless_blocks_total"), Some(1.0));
        assert_eq!(f.tables.indexed_blocks(f.row, 0), 2);
        for hash in [1, 2] {
            assert_eq!(first_block_holders(&f, hash), [(f.row, 1)], "block {hash}");
        }
    });
}

/// One store from `parent` of the block 1 and the block `hash`
/// behind it, each as wide as the block of record.
fn forked(hash: i64, parent: Option<i64>) -> KvBlocksStored {
    let wide = |hash: i64| KvBlock {
        block_size: 32,
        ..block(hash, &tokens_of(hash))
    };
    KvBlocksStored {
        blocks: vec![wide(1), wide(hash)],
        parent_block_hash: parent,
        ..KvBlocksStored::default()
    }
}

/// Blocks the index declines at a size cap are counted for the
/// worker: on a store from the root, and on the retry from the root
/// of a store whose parent the worker does not hold.
#[test]
fn blocks_declined_at_a_size_cap_are_counted_for_the_worker() {
    // The first chain's second block lengthens the first block's
    // own run. Each later chain's second block is one more entry of
    // the table behind that run, which holds 196,608 at the most.
    const CHAINS: i64 = 196_609;
    let series = Series::new();
    metrics::with_local_recorder(&series.recorder, || {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let (_host, sink) = opened(&f);
        let count = |name: &str| series.read(name, &[]);
        let stores = "smg_kv_event_cap_refused_stores_total";
        let blocks = "smg_kv_event_cap_refused_blocks_total";
        let fill = (0..CHAINS).map(|chain| Data::Stored(forked(1_000 + chain, None)));
        assert_eq!(
            sink.admit(f.row, 0, facts(1), &batch(fill)),
            BatchOutcome::Applied
        );
        assert_eq!(
            count(stores),
            None,
            "the table takes its entries up to the cap"
        );
        let held = usize::try_from(CHAINS).expect("a count of chains") + 1;
        assert_eq!(f.tables.indexed_blocks(f.row, 0), held);
        // The first block is held, and the second is declined.
        let over = batch([Data::Stored(forked(2_000_000, None))]);
        assert_eq!(sink.admit(f.row, 0, facts(2), &over), BatchOutcome::Applied);
        assert_eq!((count(stores), count(blocks)), (Some(1.0), Some(1.0)));
        // Under a parent the worker does not hold, the retry from
        // the root is declined the same way.
        let orphan = batch([Data::Stored(forked(2_000_001, Some(99)))]);
        assert_eq!(
            sink.admit(f.row, 0, facts(3), &orphan),
            BatchOutcome::Applied
        );
        assert_eq!((count(stores), count(blocks)), (Some(2.0), Some(2.0)));
        let parentless = count("smg_kv_event_parentless_stores_total");
        assert_eq!(parentless, Some(1.0));
        assert_eq!(f.tables.indexed_blocks(f.row, 0), held);
    });
}

/// A batch that holds a released state names no series. It took its
/// row's state ahead of its stream's removal and comes behind it: it is
/// passed over ahead of the row's cursor, which stays unheard.
#[test]
fn a_batch_that_holds_a_released_state_names_no_series() {
    let series = Series::new();
    metrics::with_local_recorder(&series.recorder, || {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let (_host, sink) = opened(&f);
        let feed = sink.feeds.lock().get(&f.row).cloned();
        let feed = feed.expect("the stream's feed");
        let state = sink.applied(f.row, (f.row, 0)).expect("the row's state");
        sink.removed(f.row);
        let late = wide(1, 32);
        let outcome = sink.admit_under(f.row, (f.row, 0), &feed, state, facts(1), &late);
        assert_eq!(outcome, BatchOutcome::Skipped);
        assert!(!series.names_the_worker());
        assert_eq!(f.tables.cursor(f.row, 0), None);
    });
}

/// A stream that left names no series: its batch is passed over and
/// counted nowhere, and a reset it begins then counts under no worker.
#[test]
fn a_stream_that_left_names_no_series() {
    let series = Series::new();
    metrics::with_local_recorder(&series.recorder, || {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let (_host, sink) = opened(&f);
        sink.removed(f.row);
        assert_eq!(
            sink.admit(f.row, 0, facts(1), &wide(1, 32)),
            BatchOutcome::Skipped
        );
        sink.resync(f.row, None, ResyncReason::DataLoss);
        assert!(!series.names_the_worker());
    });
}
