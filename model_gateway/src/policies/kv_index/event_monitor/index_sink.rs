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
#[path = "../../../../tests/unit/policies/kv_index/event_monitor/index_sink/tests.rs"]
mod tests;
