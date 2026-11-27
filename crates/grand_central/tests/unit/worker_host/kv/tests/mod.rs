use std::sync::mpsc;

use smg_worker::{
    common_proto::{
        KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheCleared, KvCacheEvent, KvCacheLocality,
        KvCacheTier,
    },
    identity::builder::BasicWorkerBuilder,
    kv_events::port::SnapshotChunk,
    registry::WorkerRegistry,
};

use super::*;
use crate::{
    store::{MemoryStore, Registrant, Signals, WorkerFacts, WorkersStore},
    worker_host::{objects::tests::lay_standing, Members},
};

mod application;
mod feeds_and_block_size;
mod ranks;

const MODEL: &str = "m";
const URL: &str = "http://a:8000";
const RANK: u16 = 1;
const BLOCK_SIZE: u32 = 32;

impl KvAdapter {
    /// An adapter that writes `tables`, every batch to its stream's row. It
    /// has no holder to state its rows: the cases that build one state
    /// theirs to the tables themselves.
    fn new(tables: Arc<KvTables>) -> Self {
        Self::over(tables, None)
    }
}

/// A store's rows, the tables beside them and an adapter with no
/// holder: the cases state a row to the tables themselves
/// ([`Fixture::state`]), as a holder does where it lays the row. The
/// row's feed is open on the adapter ([`open_feed`]): a stream writes
/// only while its feed stands open.
struct Fixture {
    store: Arc<MemoryStore>,
    tables: Arc<KvTables>,
    sink: Arc<KvAdapter>,
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
        let tables = Arc::new(KvTables::default());
        let sink = Arc::new(KvAdapter::new(Arc::clone(&tables)));
        let fixture = Self {
            store,
            tables,
            sink,
            row,
        };
        fixture.state(row);
        open_feed(&fixture.sink, row);
        fixture
    }

    /// State the row `id` to the tables by the facts the store holds
    /// for it now, as a holder does where it lays the row.
    fn state(&self, id: WorkerId) {
        let row = self.store.get(id).expect("the row");
        self.tables.member(&row);
    }

    fn applied(&self, sequence: u64, batch: &KvEventBatch) {
        assert_eq!(
            self.sink
                .admit(self.row, RANK, facts_of(sequence, batch), batch),
            BatchOutcome::Applied
        );
    }

    fn blocks(&self) -> usize {
        self.tables.indexed_blocks(self.row, RANK)
    }

    /// A host over a holder that was handed the fixture's tables and
    /// has laid the store's rows.
    fn host(&self) -> KvHost<Members> {
        let objects = Arc::new(
            Objects::new(Arc::new(WorkerRegistry::new())).with_kv_tables(Arc::clone(&self.tables)),
        );
        lay_standing(&objects, &self.store);
        KvHost::new(objects)
    }
}

/// Open `row`'s feed on `sink`, as a host's opening does: the row
/// begins, and the feed's record is kept.
fn open_feed(sink: &KvAdapter, row: WorkerId) {
    assert_eq!(sink.begin(row, || true), Ok(()), "the row begins");
    sink.opened(row, MODEL, URL);
}

fn facts(sequence: u64) -> BatchFacts {
    BatchFacts {
        sequence,
        ..BatchFacts::default()
    }
}

/// Whether the adapter keeps `row` as its own stream's, begun from
/// nothing: no batch was applied under its state.
fn begun(sink: &KvAdapter, row: WorkerId) -> bool {
    let rows = sink.rows.lock();
    rows.written.get(&row).is_some_and(|written| {
        written.stream == row && !written.leaving && written.applied.lock().emptied.is_none()
    })
}

/// The header's facts of `batch` at `sequence`, by the collector's rule:
/// a clear no name owns is the header's clear, and a clear a name owns
/// is not.
fn facts_of(sequence: u64, batch: &KvEventBatch) -> BatchFacts {
    let clears = batch.events.iter().any(
        |event| matches!(&event.data, Some(Data::Cleared(cleared)) if cleared.ownership.is_none()),
    );
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

/// The feed of the fixture's one worker, opened by its host.
fn opened(f: &Fixture) -> (KvHost<Members>, Arc<KvAdapter>) {
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

/// The cache-event series of the fixture's worker, read from a recorder
/// of the case's own. The recorder's crates come with the controller.
#[cfg(feature = "controller")]
mod series {
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

    /// A stream that left names no series: its batch is passed over and
    /// counted nowhere, and its reset counts under no worker.
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
}
