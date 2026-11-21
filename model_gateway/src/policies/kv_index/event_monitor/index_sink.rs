//! The index sink: the KV event port over each model's [`KvIndex`]. It holds
//! what a worker's subscription task held as a local before the port stood
//! between the two: the worker's stream state (the admission cursor of each
//! rank over the worker's index state), found by the row's id on every call
//! and gone when the worker is removed.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use dashmap::DashMap;
use kv_index::WorkerIdExhausted;
use parking_lot::Mutex;
use rustc_hash::FxBuildHasher;
use smg_grpc_client::common_proto::{kv_cache_event, KvEventBatch};
use smg_worker::{
    kv_events::apply::{store_width, WorkerIndexCounters, WorkerIndexState},
    metrics::writers::Metrics,
};
use tracing::{debug, info};

use super::{
    admission::{self, WorkerStreamState},
    port::{BatchFacts, BatchOutcome, KvCursor, KvEventSink, ResyncReason},
    KvBlockSize, KvEventMonitor,
};
use crate::policies::kv_index::index_backend::KvIndex;

/// The sink's name for one worker's row: made at [`IndexSink::open`] and
/// given back by the collector on every call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RowId(u64);

/// What the sink keeps for one worker: where its blocks go, and the state of
/// its stream.
struct Row {
    index: Arc<KvIndex>,
    /// The worker's id in `index`.
    worker_id: u32,
    worker_url: String,
    model_id: String,
    /// Locked for the length of one call. One task feeds a row, so the lock
    /// is never waited on.
    state: Mutex<WorkerStreamState>,
}

/// The sink of every events subscription of one monitor.
pub(crate) struct IndexSink {
    /// The rows by id, read on every call: the ids are the sink's own
    /// counter, so the map takes the cheap hasher.
    rows: DashMap<RowId, Arc<Row>, FxBuildHasher>,
    next_row: AtomicU64,
    /// The monitor's block sizes by model, which a stored block confirms.
    block_sizes: Arc<DashMap<String, KvBlockSize>>,
}

/// A row for the length of one subscription, and the sink its collector
/// feeds: the collector holds the row and hands it to the job that says the
/// worker left, so a task aborted while that job is pending takes nothing
/// from under it. Dropped without [`KvEventSink::removed`] (the task
/// panicked, or was aborted ahead of that job), it takes the row's state
/// with it, as the task's local went, and leaves the worker's blocks in the
/// index, as they stayed.
pub(crate) struct OpenRow {
    sink: Arc<IndexSink>,
    id: RowId,
}

impl OpenRow {
    /// The row as its collector holds it, with the row's id.
    pub(crate) fn into_feed(self) -> (Arc<Self>, RowId) {
        let id = self.id;
        (Arc::new(self), id)
    }
}

/// Every call goes to the sink the row stands in.
impl KvEventSink for OpenRow {
    type Batch = KvEventBatch;
    type Id = RowId;

    fn admit(&self, id: RowId, rank: u16, facts: BatchFacts, batch: &KvEventBatch) -> BatchOutcome {
        self.sink.admit(id, rank, facts, batch)
    }

    fn reconnected(&self, id: RowId) {
        self.sink.reconnected(id);
    }

    fn resync(&self, id: RowId, rank: Option<u16>, reason: ResyncReason) {
        self.sink.resync(id, rank, reason);
    }

    fn cursor(&self, id: RowId, rank: u16) -> Option<KvCursor> {
        self.sink.cursor(id, rank)
    }

    fn removed(&self, id: RowId) {
        self.sink.removed(id);
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.sink.block_size(model)
    }

    fn set_block_size(&self, model: &str, size: u32) {
        self.sink.set_block_size(model, size);
    }
}

impl Drop for OpenRow {
    fn drop(&mut self) {
        self.sink.rows.remove(&self.id);
    }
}

impl KvEventMonitor {
    /// Learn the model's `block_size` from the stored blocks of a batch.
    ///
    /// The first stored block replaces a provisional size (a `WorkerSpec`
    /// seed): the event stream reflects the backend's actual page size.
    /// After that a block only raises the size. Hybrid engines publish,
    /// beside their full blocks, partial entries at their hash granularity
    /// (a fraction of the block, for a prefix boundary a later request can
    /// resume at); a request chunked at that size would never match a full
    /// block, so a smaller block never lowers the size of record.
    /// `CacheAwarePolicy` uses the size to chunk request tokens into blocks
    /// for overlap scoring.
    pub(super) fn learn_block_size(
        block_sizes: &DashMap<String, KvBlockSize>,
        model_id: &str,
        batch: &KvEventBatch,
    ) {
        for event in &batch.events {
            let Some(kv_cache_event::Data::Stored(stored)) = &event.data else {
                continue;
            };
            let Some(tokens) = store_width(stored).and_then(|width| usize::try_from(width).ok())
            else {
                continue;
            };
            // The common case, a block of the size of record, costs one shared
            // read; the write lock is taken only to change the size.
            if block_sizes
                .get(model_id)
                .is_some_and(|size| size.from_events && tokens <= size.tokens)
            {
                continue;
            }
            let mut entry = block_sizes
                .entry(model_id.to_string())
                .or_insert(KvBlockSize {
                    tokens: 0,
                    from_events: false,
                });
            if entry.from_events && tokens <= entry.tokens {
                continue;
            }
            let previous = (entry.tokens > 0).then_some(entry.tokens);
            *entry = KvBlockSize {
                tokens,
                from_events: true,
            };
            Metrics::set_kv_block_size(model_id, tokens);
            info!(
                model_id = %model_id,
                block_size = tokens,
                previous,
                "Learned block_size from KV event"
            );
        }
    }
}

impl IndexSink {
    /// A sink with no row, over the monitor's block sizes.
    pub(super) fn new(block_sizes: Arc<DashMap<String, KvBlockSize>>) -> Self {
        Self {
            rows: DashMap::with_hasher(FxBuildHasher),
            next_row: AtomicU64::new(0),
            block_sizes,
        }
    }

    /// Intern the worker in its model's `index` and open a row with nothing
    /// in it: no cursor, no block. `rank` is the rank the worker stands for,
    /// `None` for its whole engine.
    pub(super) fn open(
        self: &Arc<Self>,
        index: Arc<KvIndex>,
        model_id: &str,
        worker_url: &str,
        rank: Option<usize>,
    ) -> Result<OpenRow, WorkerIdExhausted> {
        let worker_id = index.intern_worker(worker_url)?;
        let id = RowId(self.next_row.fetch_add(1, Ordering::Relaxed));
        self.rows.insert(
            id,
            Arc::new(Row {
                index,
                worker_id,
                worker_url: worker_url.to_string(),
                model_id: model_id.to_string(),
                state: Mutex::new(WorkerStreamState::of_rank(rank)),
            }),
        );
        Ok(OpenRow {
            sink: Arc::clone(self),
            id,
        })
    }

    /// The row under `id`, held past the map's own lock.
    fn row(&self, id: RowId) -> Option<Arc<Row>> {
        self.rows.get(&id).map(|row| Arc::clone(row.value()))
    }
}

impl KvEventSink for IndexSink {
    type Batch = KvEventBatch;
    type Id = RowId;

    /// `rank` is not read: the batch is the wire message and names its own
    /// rank, or none, and the row's cursors are keyed by that.
    fn admit(
        &self,
        id: RowId,
        _rank: u16,
        facts: BatchFacts,
        batch: &KvEventBatch,
    ) -> BatchOutcome {
        let Some(row) = self.row(id) else {
            return BatchOutcome::Skipped;
        };
        let mut state = row.state.lock();
        // The admission calls this once the cursor took the batch, before
        // the batch is applied: a stored block of it confirms the model's
        // block size there.
        let mut taken = false;
        let mut on_taken = |batch: &KvEventBatch| {
            taken = true;
            KvEventMonitor::learn_block_size(&self.block_sizes, &row.model_id, batch);
        };
        let outcome = KvEventMonitor::admit_batch(
            batch,
            facts,
            &row.worker_url,
            row.worker_id,
            &row.index,
            &mut state,
            &mut on_taken,
        );
        match outcome {
            admission::BatchOutcome::Gap { expected, received } => {
                BatchOutcome::Gap { expected, received }
            }
            _ if taken => BatchOutcome::Applied,
            _ => BatchOutcome::Skipped,
        }
    }

    fn reconnected(&self, id: RowId) {
        if let Some(row) = self.row(id) {
            row.state.lock().reconnected();
        }
    }

    fn resync(&self, id: RowId, rank: Option<u16>, reason: ResyncReason) {
        if reason == ResyncReason::Snapshot {
            // A snapshot's start, said ahead of its first chunk. This sink
            // reads the start in that chunk's own batch, where it clears the
            // worker and counts the resync: the word ahead changes nothing.
            return;
        }
        let Some(row) = self.row(id) else {
            return;
        };
        let mut state = row.state.lock();
        if let (Some(asked), Some(mine)) = (rank, state.rank) {
            if i32::from(asked) != mine {
                return;
            }
        }
        if reason == ResyncReason::SnapshotAbandoned {
            // The snapshot stream cannot be trusted: the cursors start over,
            // so the next subscription asks from zero, and the blocks stay
            // until the next snapshot's first chunk clears them. No resync is
            // counted for it.
            state.abandon_snapshot();
            return;
        }
        KvEventMonitor::reset_worker(
            &row.index,
            row.worker_id,
            &mut state,
            &row.worker_url,
            reason,
        );
    }

    fn cursor(&self, id: RowId, rank: u16) -> Option<KvCursor> {
        let row = self.row(id)?;
        let state = row.state.lock();
        let cursor = state.ranks.get(&i32::from(rank))?;
        Some(KvCursor {
            applied: cursor.resume_from(),
            replay_pending: cursor.pending_gap() > 0,
            degraded: cursor.is_degraded(),
        })
    }

    /// The caller runs this off the async runtime: the worker's blocks leave
    /// the index inside the call.
    fn removed(&self, id: RowId) {
        let Some((_, row)) = self.rows.remove(&id) else {
            return;
        };
        let state = std::mem::take(&mut *row.state.lock());
        let WorkerIndexState {
            blocks, counters, ..
        } = state.index;
        if counters != WorkerIndexCounters::default() {
            debug!(
                worker_id = row.worker_id,
                ?counters,
                "KV events the index did not take as is"
            );
        }
        row.index.remove_worker(row.worker_id, blocks);
        Metrics::set_kv_index_blocks(&row.worker_url, 0);
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.block_sizes
            .get(model)
            .and_then(|size| u32::try_from(size.tokens).ok())
    }

    /// A registration's size stands only where none is known: the first
    /// stored block replaces it, and it never replaces a size the events
    /// named.
    fn set_block_size(&self, model: &str, size: u32) {
        let known = self
            .block_sizes
            .entry(model.to_string())
            .or_insert(KvBlockSize {
                tokens: usize::try_from(size).unwrap_or(usize::MAX),
                from_events: false,
            });
        // Under the entry's lock, as the learn site writes it, so a size a
        // stream learns meanwhile is not overwritten on the gauge by the
        // seed.
        Metrics::set_kv_block_size(model, known.tokens);
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::common_proto::{KvBlock, KvBlocksStored, KvCacheEvent, KvSnapshotChunk};
    use smg_worker::kv_events::{batch_facts, batch_rank};

    use super::*;

    /// One stored block of sixteen tokens that start at `first_token`, as
    /// batch `sequence` of `rank`.
    fn stored(sequence: u64, first_token: u32, rank: Option<i32>) -> KvEventBatch {
        KvEventBatch {
            sequence_number: sequence,
            dp_rank: rank,
            events: vec![KvCacheEvent {
                event_id: sequence,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![KvBlock {
                        block_hash: i64::from(first_token),
                        token_ids: (first_token..first_token + 16).collect(),
                        block_size: 16,
                        ..Default::default()
                    }],
                    ..Default::default()
                })),
            }],
            ..Default::default()
        }
    }

    /// A sink over one fresh index, with a row open for a worker that
    /// stands for `rank`.
    fn opened(rank: Option<usize>) -> (Arc<KvIndex>, Arc<IndexSink>, OpenRow, RowId) {
        let index = Arc::new(KvIndex::chain());
        let sink = Arc::new(IndexSink::new(Arc::new(DashMap::new())));
        let row = sink
            .open(Arc::clone(&index), "m", "grpc://w1:9000", rank)
            .expect("the index interns the worker");
        let id = row.id;
        (index, sink, row, id)
    }

    /// Say `batch` to the sink as the collector does.
    fn admit(sink: &IndexSink, id: RowId, batch: &KvEventBatch) -> BatchOutcome {
        let rank = batch_rank(batch).expect("the case's rank is a rank");
        sink.admit(id, rank, batch_facts(batch), batch)
    }

    fn applied(sink: &IndexSink, id: RowId, rank: u16) -> Option<u64> {
        sink.cursor(id, rank).map(|cursor| cursor.applied)
    }

    /// A worker that left leaves nothing under its id: no cursor, no block,
    /// no row. Added again it is a new row that starts from nothing: its
    /// first batch is taken wherever it stands, a sequence the old cursor
    /// had passed included.
    #[test]
    fn a_removed_worker_leaves_no_cursor_and_starts_from_nothing_when_added_again() {
        let (index, sink, _row, id) = opened(None);
        assert_eq!(
            admit(&sink, id, &stored(7, 1_000, None)),
            BatchOutcome::Applied
        );
        assert_eq!(
            admit(&sink, id, &stored(8, 2_000, None)),
            BatchOutcome::Applied
        );
        assert_eq!(applied(&sink, id, 0), Some(8));
        assert_ne!(index.current_size(), 0);

        sink.removed(id);
        assert_eq!(sink.cursor(id, 0), None, "no cursor is left under the id");
        assert_eq!(sink.rows.len(), 0);
        assert_eq!(index.current_size(), 0, "its blocks left the index");
        // What is said for the id after the worker left reaches nothing.
        assert_eq!(
            admit(&sink, id, &stored(9, 3_000, None)),
            BatchOutcome::Skipped
        );
        sink.reconnected(id);
        sink.resync(id, None, ResyncReason::DataLoss);
        assert_eq!(index.current_size(), 0);

        let again = sink
            .open(Arc::clone(&index), "m", "grpc://w1:9000", None)
            .expect("the index interns the worker again");
        let next = again.id;
        assert_ne!(next, id);
        assert_eq!(sink.cursor(next, 0), None, "the new row has heard no rank");
        // Sequence 3 is below the old cursor: a row that had kept it would
        // drop the batch as a duplicate.
        assert_eq!(
            admit(&sink, next, &stored(3, 4_000, None)),
            BatchOutcome::Applied
        );
        assert_eq!(applied(&sink, next, 0), Some(3));
    }

    /// The sink's cursor is what a subscription resumes from. It stands at
    /// the last sequence applied across a new connection, shows a replay
    /// asked for, and starts over, with the worker's blocks gone, when the
    /// server's history is lost.
    #[test]
    fn the_cursor_stands_across_a_reconnect_and_starts_over_at_a_resync() {
        let (index, sink, _row, id) = opened(None);
        assert_eq!(
            admit(&sink, id, &stored(7, 1_000, None)),
            BatchOutcome::Applied
        );
        assert_eq!(
            admit(&sink, id, &stored(8, 2_000, None)),
            BatchOutcome::Applied
        );
        sink.reconnected(id);
        assert_eq!(applied(&sink, id, 0), Some(8));

        assert_eq!(
            admit(&sink, id, &stored(11, 3_000, None)),
            BatchOutcome::Gap {
                expected: 9,
                received: 11
            }
        );
        assert_eq!(
            sink.cursor(id, 0),
            Some(KvCursor {
                applied: 8,
                replay_pending: true,
                degraded: false
            })
        );

        sink.resync(id, None, ResyncReason::OutOfRange);
        assert_eq!(sink.cursor(id, 0), Some(KvCursor::default()));
        assert_eq!(index.current_size(), 0);
    }

    /// A resync names a rank as a check: for a rank the row does not stand
    /// for, it resets nothing. An abandoned snapshot starts the cursors over
    /// and keeps the blocks its chunks brought.
    #[test]
    fn a_resync_checks_its_rank_and_an_abandoned_snapshot_keeps_its_blocks() {
        let (index, sink, _row, id) = opened(Some(1));
        assert_eq!(
            admit(&sink, id, &stored(4, 1_000, Some(1))),
            BatchOutcome::Applied
        );
        sink.resync(id, Some(0), ResyncReason::DataLoss);
        assert_eq!(applied(&sink, id, 1), Some(4), "rank 0 is not the row's");
        assert_ne!(index.current_size(), 0);

        let chunk = KvEventBatch {
            snapshot: Some(KvSnapshotChunk {
                index: 0,
                count: 2,
                ..Default::default()
            }),
            ..stored(9, 2_000, Some(1))
        };
        assert_eq!(admit(&sink, id, &chunk), BatchOutcome::Applied);
        assert_eq!(applied(&sink, id, 1), Some(9));
        let blocks = index.current_size();
        assert_ne!(blocks, 0);
        sink.resync(id, Some(1), ResyncReason::SnapshotAbandoned);
        assert_eq!(sink.cursor(id, 1), Some(KvCursor::default()));
        assert_eq!(index.current_size(), blocks, "the chunk's blocks stay");

        sink.resync(id, Some(1), ResyncReason::DataLoss);
        assert_eq!(index.current_size(), 0);
    }

    /// Malformed-snapshot recovery starts every cursor over even when the
    /// sink already counted the preceding snapshot as complete.
    #[test]
    fn an_abandoned_snapshot_resets_cursors_after_the_sink_closed_its_snapshot() {
        let (index, sink, _row, id) = opened(None);
        let chunk = KvEventBatch {
            snapshot: Some(KvSnapshotChunk {
                index: 0,
                count: 1,
                ..Default::default()
            }),
            ..stored(6, 1_000, None)
        };
        assert_eq!(admit(&sink, id, &chunk), BatchOutcome::Applied);
        assert_eq!(applied(&sink, id, 0), Some(6));
        let blocks = index.current_size();
        assert_ne!(blocks, 0);

        sink.resync(id, None, ResyncReason::SnapshotAbandoned);

        assert_eq!(sink.cursor(id, 0), Some(KvCursor::default()));
        assert_eq!(
            index.current_size(),
            blocks,
            "the completed chunk's blocks stay"
        );
    }

    /// A snapshot's start said ahead of its first chunk changes nothing by
    /// itself: this sink clears in the chunk's own batch.
    #[test]
    fn a_snapshots_start_said_ahead_changes_nothing_by_itself() {
        let (index, sink, _row, id) = opened(None);
        assert_eq!(
            admit(&sink, id, &stored(4, 1_000, None)),
            BatchOutcome::Applied
        );
        let blocks = index.current_size();
        assert_ne!(blocks, 0);
        sink.resync(id, None, ResyncReason::Snapshot);
        assert_eq!(applied(&sink, id, 0), Some(4));
        assert_eq!(index.current_size(), blocks);
    }

    /// A row dropped without the worker's removal (its task panicked) takes
    /// its state with it and leaves the worker's blocks in the index.
    #[test]
    fn a_row_dropped_without_a_removal_leaves_its_blocks() {
        let (index, sink, row, id) = opened(None);
        assert_eq!(
            admit(&sink, id, &stored(1, 1_000, None)),
            BatchOutcome::Applied
        );
        drop(row);
        assert_eq!(sink.rows.len(), 0);
        assert_eq!(sink.cursor(id, 0), None);
        assert_ne!(index.current_size(), 0);
    }

    /// The job that says a worker left holds the row itself: a task aborted
    /// while the job is pending drops nothing the job needs, and the
    /// worker's blocks leave the index. A row the task alone held went with
    /// the task, ahead of the job, which then found nothing to remove.
    #[test]
    fn a_dispatched_removal_holds_its_row_past_its_task() {
        let (index, sink, row, id) = opened(None);
        assert_eq!(
            admit(&sink, id, &stored(1, 1_000, None)),
            BatchOutcome::Applied
        );
        // The collector's hold moves into the job at its dispatch, and the
        // task is gone before the job runs.
        let (feed, id) = row.into_feed();
        std::thread::spawn(move || feed.removed(id))
            .join()
            .expect("the job runs");
        assert_eq!(index.current_size(), 0);
        assert_eq!(sink.rows.len(), 0, "the row went with the job's hold");
    }

    /// The outcome is read by the cursor. A batch of a rank the row does not
    /// index advances that rank's cursor: it is `Applied`, with nothing
    /// indexed, and a stored block of it confirms the model's block size. A
    /// duplicate below the cursor is `Skipped`.
    #[test]
    fn the_outcome_says_whether_the_cursor_took_the_batch() {
        let (index, sink, _row, id) = opened(Some(1));
        let other = stored(5, 1_000, Some(0));
        assert_eq!(admit(&sink, id, &other), BatchOutcome::Applied);
        assert_eq!(index.current_size(), 0, "rank 0's block is not the row's");
        assert_eq!(applied(&sink, id, 0), Some(5));
        assert_eq!(sink.block_size("m"), Some(16));
        assert_eq!(admit(&sink, id, &other), BatchOutcome::Skipped);
    }

    /// A registration's block size stands until a stored block names one,
    /// and never replaces a size the events named.
    #[test]
    fn a_registrations_block_size_is_provisional() {
        let (_index, sink, _row, id) = opened(None);
        assert_eq!(sink.block_size("m"), None);
        sink.set_block_size("m", 64);
        assert_eq!(sink.block_size("m"), Some(64));
        assert_eq!(
            admit(&sink, id, &stored(1, 1_000, None)),
            BatchOutcome::Applied
        );
        assert_eq!(sink.block_size("m"), Some(16));
        sink.set_block_size("m", 64);
        assert_eq!(sink.block_size("m"), Some(16));
    }
}
