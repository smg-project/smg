//! One stream's state: the admission cursor per data-parallel rank, gap
//! recovery, relay snapshots, and the metrics around an applied batch.

use std::{collections::HashMap, time::Duration};

use smg_grpc_client::common_proto::{KvEventBatch, KvSnapshotChunk};
use smg_worker::{
    identity::endpoint::Endpoint,
    kv_events::apply::{apply_cleared, apply_counted, WorkerIndexState},
    metrics::writers::Metrics,
};
use tracing::{debug, info, warn};

use super::KvEventMonitor;
use crate::policies::kv_index::{
    event_monitor::{
        port::{BatchFacts, ResyncReason},
        recovery::{Admission, RankState},
    },
    index_backend::{KvIndex, WorkerBlocks},
};

/// A worker's subscription state: one admission cursor per data-parallel
/// rank (every publisher numbers its own batches) over the worker's one
/// index state, whose copies are pooled across ranks (see
/// [`WorkerIndexState`]). Cursors and copy counts never mix.
#[derive(Default)]
pub(super) struct WorkerStreamState {
    pub(super) ranks: HashMap<i32, RankState>,
    pub(super) index: WorkerIndexState<WorkerBlocks>,
    /// A relay snapshot whose chunks are still arriving on the stream.
    pub(super) snapshot: Option<SnapshotProgress>,
    /// The data-parallel rank this worker stands for, where it is a rank
    /// worker: `None` for a worker that stands for its whole engine. An engine's
    /// stream names every rank's batches: all of them move their rank's
    /// cursor and the snapshot bookkeeping, only this rank's (and the
    /// unranked, which are every rank's) enter the index or clear it.
    pub(super) rank: Option<i32>,
    /// Whether a batch that names no rank has ridden rank 0's cursor: such a
    /// batch is every rank's and is indexed by this worker, so that cursor
    /// then counts toward this worker's degraded ranks even when the worker
    /// stands for another rank.
    pub(super) saw_unranked: bool,
}

/// Where an in-band relay snapshot stands (`KvSnapshotChunk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SnapshotProgress {
    count: u32,
    applied: u32,
    blocks: u64,
}

impl WorkerStreamState {
    /// The state of a worker that stands for `rank` of its engine (`None`:
    /// the whole engine).
    pub(super) fn of_rank(rank: Option<usize>) -> Self {
        Self {
            rank: rank.and_then(|rank| i32::try_from(rank).ok()),
            ..Self::default()
        }
    }

    /// Whether a batch's blocks are this worker's cache: its own rank's, or
    /// unranked (every rank's). Another rank's are counted, not indexed.
    fn indexes(&self, batch: &KvEventBatch) -> bool {
        match (self.rank, batch.dp_rank) {
            (Some(mine), Some(theirs)) => mine == theirs,
            _ => true,
        }
    }

    /// The cursor to send when resubscribing: rank 0's last applied sequence
    /// (the servicers replay rank 0's publisher). Other ranks keep their own
    /// cursors and dedup what arrives.
    #[cfg(test)]
    pub(super) fn resume_sequence(&self) -> u64 {
        self.ranks.get(&0).map_or(0, RankState::resume_from)
    }

    /// A new stream connection was made: every rank's next batch is its
    /// first on it.
    pub(super) fn reconnected(&mut self) {
        for rank in self.ranks.values_mut() {
            rank.reconnected();
        }
    }

    /// The degraded cursors among the ranks this worker indexes: every rank
    /// of an unpinned worker; for a rank worker its own rank, plus rank 0's
    /// cursor once a batch that names no rank (every rank's, indexed here)
    /// has ridden it. Another rank's unreplayed gap says nothing about this
    /// worker's index.
    pub(super) fn degraded_ranks(&self) -> usize {
        self.ranks
            .iter()
            .filter(|(rank, cursor)| self.counts_rank(**rank) && cursor.is_degraded())
            .count()
    }

    /// Whether `rank`'s cursor carries batches this worker indexes.
    fn counts_rank(&self, rank: i32) -> bool {
        match self.rank {
            None => true,
            Some(mine) => mine == rank || (rank == 0 && self.saw_unranked),
        }
    }

    /// A stream's snapshot was abandoned: start every cursor over so the next
    /// subscription asks from zero, whether or not this sink still considers
    /// the snapshot open. Returns whether a snapshot was open.
    pub(super) fn abandon_snapshot(&mut self) -> bool {
        for cursor in self.ranks.values_mut() {
            cursor.reset();
        }
        let progress = self.snapshot.take();
        if let Some(progress) = progress {
            debug_assert!(progress.applied < progress.count);
        }
        progress.is_some()
    }
}

/// What `admit_batch` did with a batch.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum BatchOutcome {
    /// Applied to the index.
    Applied,
    /// Dropped (a duplicate), held (a snapshot tail) or another rank's
    /// (counted, not indexed).
    Skipped,
    /// A gap: the caller reconnects asking for a replay from `expected`.
    Gap { expected: u64, received: u64 },
}

impl KvEventMonitor {
    /// Run one batch through its rank's admission cursor and apply it to the
    /// worker's index state. `facts` are the batch's header as the collector
    /// read it: the cursor decides by them.
    pub(super) fn admit_batch(
        batch: &KvEventBatch,
        facts: BatchFacts,
        worker_url: &str,
        worker_id: u32,
        indexer: &KvIndex,
        state: &mut WorkerStreamState,
        on_batch: &mut impl FnMut(&KvEventBatch),
    ) -> BatchOutcome {
        if let Some(chunk) = &batch.snapshot {
            return Self::admit_snapshot_chunk(
                batch, chunk, facts, worker_url, worker_id, indexer, state, on_batch,
            );
        }
        let rank = batch.dp_rank.unwrap_or(0);
        // The first batch that names no rank puts rank 0's cursor into a
        // pinned worker's degraded count: a change of the gauge in its own
        // right, republished below with the rest.
        let mut degraded_changed = false;
        if batch.dp_rank.is_none() {
            degraded_changed |= !state.saw_unranked;
            state.saw_unranked = true;
        }
        let seq = facts.sequence;
        let clears = facts.clears;
        // Whether the batch's rank is one this worker indexes: its clear
        // reaches the index only then (another rank's restart or cleared gap
        // says nothing about this worker's blocks).
        let indexed = state.indexes(batch);
        let cursor = state.ranks.entry(rank).or_default();
        // Read before `admit` can drop a pending replay: a restart the gateway
        // notices here loses the replay's batches as a refused resume does.
        let pending = cursor.pending_gap();
        let admission = cursor.admit(seq, clears);
        match admission {
            Admission::Apply => {}
            Admission::Stale => {
                debug!(worker_url = %Endpoint::redacted_str(worker_url), rank, received = seq, "Skipping stale KV event batch");
                Metrics::record_kv_event_batch(worker_url, "stale");
                return BatchOutcome::Skipped;
            }
            Admission::Restart if indexed => {
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    rank,
                    received = seq,
                    "KV event publisher restarted; clearing the worker's index state"
                );
                let dropped = Self::clear_worker(worker_id, indexer, state, rank);
                // A replay that was pending went with the old publisher, on
                // this rank or on another whose cursor the clear reset: its
                // batches count as missed here, once, as they do when the
                // relay reports the restart as `DATA_LOSS` (`reset_worker`).
                Metrics::record_kv_event_missed_batches(worker_url, pending + dropped);
                Metrics::record_kv_event_resync(
                    worker_url,
                    ResyncReason::PublisherRestart.as_str(),
                );
                degraded_changed = true;
            }
            Admission::Restart => {
                // Another rank's publisher restarted: its cursor started over
                // in `admit`, and this worker's index holds none of its
                // blocks, so nothing is cleared and the other cursors keep
                // their place (a whole-engine restart clears a rank worker
                // once, through its own rank's restart). A replay pending on
                // that rank went with its old publisher.
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    rank,
                    received = seq,
                    "KV event publisher of another rank restarted; this worker's index is kept"
                );
                Metrics::record_kv_event_missed_batches(worker_url, pending);
                degraded_changed = true;
            }
            Admission::Replay { expected } => {
                // A sighting, not a loss: the replay may bring every batch of
                // the gap, so nothing is counted missed at the request. The
                // loss is counted once where it is known: on the unrecovered
                // outcome, or where a pending replay is dropped (`reset_worker`
                // when the resume is refused, `clear_worker` when a publisher
                // starts over or a gap is cleared, on this rank or another).
                Metrics::record_kv_event_gap(worker_url, "replay_requested", 0);
                return BatchOutcome::Gap {
                    expected,
                    received: seq,
                };
            }
            Admission::Unrecovered { missed, cleared } => {
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    rank,
                    missed,
                    cleared,
                    "KV event gap could not be replayed; continuing from the live stream"
                );
                if cleared && indexed {
                    // The other ranks' cursors start over with the index: a
                    // replay pending on one of them counts as missed here,
                    // beside this rank's own unrecovered gap below. Another
                    // rank's cleared gap leaves this worker's index: it holds
                    // none of that rank's blocks.
                    let dropped = Self::clear_worker(worker_id, indexer, state, rank);
                    Metrics::record_kv_event_missed_batches(worker_url, dropped);
                    Metrics::record_kv_event_resync(worker_url, ResyncReason::GapCleared.as_str());
                }
                Metrics::record_kv_event_gap(
                    worker_url,
                    if cleared {
                        "unrecovered_cleared"
                    } else {
                        "unrecovered_kept"
                    },
                    missed,
                );
                degraded_changed = true;
            }
        }

        on_batch(batch);
        if !indexed {
            // Another rank's batch: its cursor moved and the stream is
            // working, but its blocks are that rank's cache, not this
            // worker's. Counted, not indexed.
            Metrics::record_kv_event_batch(worker_url, "other_rank");
            if degraded_changed {
                Metrics::set_kv_event_degraded_ranks(worker_url, state.degraded_ranks());
            }
            return BatchOutcome::Skipped;
        }
        let took = Self::apply_indexed_batch(
            batch, worker_url, worker_id, indexer, state, "stored", "applied",
        );
        Metrics::record_kv_event_apply(worker_url, took.as_secs_f64());
        if degraded_changed {
            Metrics::set_kv_event_degraded_ranks(worker_url, state.degraded_ranks());
        }
        BatchOutcome::Applied
    }

    /// A chunk of a relay state snapshot (`KvSnapshotChunk`): the live set the
    /// relay recorded, replacing the worker's state. Chunk 0 clears the worker
    /// and every cursor and counts a `snapshot` resync; every chunk is applied
    /// outside the admission rules and moves its rank's cursor to its stamp,
    /// which the relay chose so that live events continue after the last one.
    /// The chunk's blocks are accounted like an applied batch's: counted under
    /// `op="snapshot"` (they are the relay's record, not stores the publisher
    /// sent) and reflected in the worker's index-blocks gauge.
    #[expect(clippy::too_many_arguments, reason = "hot-path plumbing, not state")]
    fn admit_snapshot_chunk(
        batch: &KvEventBatch,
        chunk: &KvSnapshotChunk,
        facts: BatchFacts,
        worker_url: &str,
        worker_id: u32,
        indexer: &KvIndex,
        state: &mut WorkerStreamState,
        on_batch: &mut impl FnMut(&KvEventBatch),
    ) -> BatchOutcome {
        let rank = batch.dp_rank.unwrap_or(0);
        if batch.dp_rank.is_none() {
            state.saw_unranked = true;
        }
        // Read once, from the facts the collector said: the first chunk's
        // line and the cursor's mark stand on one answer.
        let unknown_before = facts.snapshot.is_some_and(|said| said.unknown_before);
        if chunk.index == 0 || state.snapshot.is_none() {
            if chunk.index != 0 {
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    rank,
                    index = chunk.index,
                    "KV event relay snapshot arrived without its first chunk; taking it as a resync"
                );
            }
            info!(
                worker_url = %Endpoint::redacted_str(worker_url),
                rank,
                chunks = chunk.count,
                blocks = chunk.blocks,
                unknown_before = chunk.unknown_before,
                through = batch.sequence_number + u64::from(chunk.count.saturating_sub(chunk.index + 1)),
                "KV event relay served a state snapshot; replacing the worker's index state"
            );
            if unknown_before {
                warn!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    rank,
                    unknown_before = chunk.unknown_before,
                    "KV event relay snapshot starts late: the engine's blocks from before the \
                     relay's record are unknown; the worker's index is partial until they leave \
                     the engine (rank degraded)"
                );
            }
            apply_cleared(worker_id, indexer, &mut state.index);
            for cursor in state.ranks.values_mut() {
                cursor.reset();
            }
            Metrics::record_kv_event_resync(worker_url, ResyncReason::Snapshot.as_str());
            state.snapshot = Some(SnapshotProgress {
                count: chunk.count.max(1),
                applied: 0,
                blocks: chunk.blocks,
            });
        }
        let cursor = state.ranks.entry(rank).or_default();
        cursor.resync_to(facts.sequence);
        if unknown_before {
            cursor.mark_degraded();
        }
        on_batch(batch);
        // Another rank's chunk takes its share of the snapshot's bookkeeping
        // (the clear above, its rank's cursor, the count below) and none of
        // its blocks: they are that rank's cache, not this worker's.
        let indexed = state.indexes(batch);
        if indexed {
            Self::apply_indexed_batch(
                batch, worker_url, worker_id, indexer, state, "snapshot", "snapshot",
            );
        } else {
            Metrics::record_kv_event_batch(worker_url, "other_rank");
        }
        if let Some(progress) = &mut state.snapshot {
            progress.applied += 1;
            if progress.applied >= progress.count {
                info!(
                    worker_url = %Endpoint::redacted_str(worker_url),
                    chunks = progress.count,
                    blocks = progress.blocks,
                    through = batch.sequence_number,
                    "KV event relay snapshot applied; continuing with live events"
                );
                state.snapshot = None;
                Metrics::set_kv_event_degraded_ranks(worker_url, state.degraded_ranks());
            }
        }
        if indexed {
            BatchOutcome::Applied
        } else {
            BatchOutcome::Skipped
        }
    }

    /// Apply a batch this worker indexes and account for it by the worker
    /// crate's one routine ([`apply_counted`]), then set the worker's
    /// index-blocks gauge. Returns how long the apply took; the live stream
    /// publishes it as its pace, a snapshot chunk does not.
    fn apply_indexed_batch(
        batch: &KvEventBatch,
        worker_url: &str,
        worker_id: u32,
        indexer: &KvIndex,
        state: &mut WorkerStreamState,
        stored_op: &'static str,
        disposition: &'static str,
    ) -> Duration {
        let took = apply_counted(
            batch,
            worker_url,
            worker_id,
            indexer,
            &mut state.index,
            stored_op,
            disposition,
        );
        Metrics::set_kv_index_blocks(worker_url, indexer.worker_block_count(worker_id));
        took
    }

    /// One rank's publisher lost its history (a restart, or an unreplayable
    /// gap too large to keep): the worker's pooled index state goes with it,
    /// and the other ranks' cursors start over so their next batch is taken
    /// as a first one instead of clearing the index a second time. A replay
    /// pending on one of those ranks goes with its cursor; the batches it was
    /// asked for are returned for the caller to count as missed.
    fn clear_worker(
        worker_id: u32,
        indexer: &KvIndex,
        state: &mut WorkerStreamState,
        rank: i32,
    ) -> u64 {
        apply_cleared(worker_id, indexer, &mut state.index);
        let mut dropped = 0;
        for (other, cursor) in &mut state.ranks {
            if *other != rank {
                dropped += cursor.pending_gap();
                cursor.reset();
            }
        }
        dropped
    }

    /// The server declared its history gone: drop the worker's index state
    /// and every cursor, so the next stream is taken from wherever it starts.
    /// A replay that was pending is lost with them: the batches it was asked
    /// for count as missed, once, beside the resync.
    pub(super) fn reset_worker(
        indexer: &KvIndex,
        worker_id: u32,
        state: &mut WorkerStreamState,
        worker_url: &str,
        reason: ResyncReason,
    ) {
        apply_cleared(worker_id, indexer, &mut state.index);
        let pending: u64 = state.ranks.values().map(RankState::pending_gap).sum();
        for cursor in state.ranks.values_mut() {
            cursor.reset();
        }
        state.snapshot = None;
        Metrics::record_kv_event_missed_batches(worker_url, pending);
        Metrics::record_kv_event_resync(worker_url, reason.as_str());
        Metrics::set_kv_event_degraded_ranks(worker_url, 0);
        Metrics::set_kv_index_blocks(worker_url, indexer.worker_block_count(worker_id));
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/policies/kv_index/event_monitor/admission/tests.rs"]
mod tests;
