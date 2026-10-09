//! One stream's state: the admission cursor per data-parallel rank, gap
//! recovery, relay snapshots, and the metrics around an applied batch.

use std::{
    collections::HashMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use smg_grpc_client::common_proto::{kv_cache_event, KvEventBatch, KvSnapshotChunk};
use tracing::{debug, info, warn};

use super::{
    apply::{WorkerIndexCounters, WorkerIndexState},
    KvEventMonitor,
};
use crate::{
    observability::metrics::Metrics,
    worker::{
        kv_event_recovery::{Admission, RankState, ResyncReason},
        kv_index_backend::KvIndex,
    },
};

/// A worker's subscription state: one admission cursor per data-parallel
/// rank (every publisher numbers its own batches) over the worker's one
/// index state, whose copies are pooled across ranks (see
/// [`WorkerIndexState`]). Cursors and copy counts never mix.
#[derive(Default)]
pub(super) struct WorkerStreamState {
    pub(super) ranks: HashMap<i32, RankState>,
    pub(super) index: WorkerIndexState,
    /// A relay snapshot whose chunks are still arriving on the stream.
    pub(super) snapshot: Option<SnapshotProgress>,
}

/// Where an in-band relay snapshot stands (`KvSnapshotChunk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SnapshotProgress {
    count: u32,
    applied: u32,
    blocks: u64,
}

impl WorkerStreamState {
    /// The cursor to send when resubscribing: rank 0's last applied sequence
    /// (the servicers replay rank 0's publisher). Other ranks keep their own
    /// cursors and dedup what arrives.
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

    pub(super) fn degraded_ranks(&self) -> usize {
        self.ranks
            .values()
            .filter(|rank| rank.is_degraded())
            .count()
    }

    /// The stream ended with a snapshot still arriving: what was applied is
    /// a partial live set, so the next subscription asks from zero (and gets
    /// a whole snapshot) instead of resuming after the last chunk's stamp.
    /// Returns whether a snapshot was abandoned.
    pub(super) fn abandon_snapshot(&mut self) -> bool {
        let Some(progress) = self.snapshot.take() else {
            return false;
        };
        for cursor in self.ranks.values_mut() {
            cursor.reset();
        }
        debug_assert!(progress.applied < progress.count);
        true
    }
}

/// What `admit_batch` did with a batch.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum BatchOutcome {
    /// Applied to the index.
    Applied,
    /// Dropped (duplicate) or held (snapshot tail).
    Skipped,
    /// A gap: the caller reconnects asking for a replay from `expected`.
    Gap { expected: u64, received: u64 },
}

impl KvEventMonitor {
    /// Run one batch through its rank's admission cursor and apply it to the
    /// worker's index state.
    pub(super) fn admit_batch(
        batch: &KvEventBatch,
        worker_url: &str,
        worker_id: u32,
        indexer: &KvIndex,
        state: &mut WorkerStreamState,
        on_batch: &mut impl FnMut(&KvEventBatch),
    ) -> BatchOutcome {
        if let Some(chunk) = &batch.snapshot {
            return Self::admit_snapshot_chunk(
                batch, chunk, worker_url, worker_id, indexer, state, on_batch,
            );
        }
        let rank = batch.dp_rank.unwrap_or(0);
        let seq = batch.sequence_number;
        let clears = batch.events.iter().any(|event| {
            matches!(&event.data, Some(kv_cache_event::Data::Cleared(cleared)) if cleared.ownership.is_none())
        });
        let admission = state.ranks.entry(rank).or_default().admit(seq, clears);
        let mut degraded_changed = false;
        match admission {
            Admission::Apply => {}
            Admission::Stale => {
                debug!(worker_url = %worker_url, rank, received = seq, "Skipping stale KV event batch");
                Metrics::record_kv_event_batch(worker_url, "stale");
                return BatchOutcome::Skipped;
            }
            Admission::Restart => {
                warn!(
                    worker_url = %worker_url,
                    rank,
                    received = seq,
                    "KV event publisher restarted; clearing the worker's index state"
                );
                Self::clear_worker(worker_id, indexer, state, rank);
                Metrics::record_kv_event_resync(
                    worker_url,
                    ResyncReason::PublisherRestart.as_str(),
                );
                degraded_changed = true;
            }
            Admission::Replay { expected } => {
                Metrics::record_kv_event_gap(worker_url, "replay_requested", seq - expected);
                return BatchOutcome::Gap {
                    expected,
                    received: seq,
                };
            }
            Admission::Unrecovered { missed, cleared } => {
                warn!(
                    worker_url = %worker_url,
                    rank,
                    missed,
                    cleared,
                    "KV event gap could not be replayed; continuing from the live stream"
                );
                if cleared {
                    Self::clear_worker(worker_id, indexer, state, rank);
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
            Admission::Buffered => {
                if let Some(cursor) = state.ranks.get_mut(&rank) {
                    if !cursor.buffer_live(batch.clone()) {
                        Metrics::record_kv_event_batch(worker_url, "tail_overflow");
                    }
                    Metrics::set_kv_event_tail_depth(worker_url, cursor.tail_len());
                }
                return BatchOutcome::Skipped;
            }
        }

        on_batch(batch);
        let started = Instant::now();
        let counters_before = state.index.counters;
        let (mut stored_blocks, mut removed_blocks) = (0usize, 0usize);
        for event in &batch.events {
            match &event.data {
                Some(kv_cache_event::Data::Stored(stored)) => stored_blocks += stored.blocks.len(),
                Some(kv_cache_event::Data::Removed(removed)) => {
                    removed_blocks += removed.block_hashes.len();
                }
                _ => {}
            }
            Self::apply_event(event, worker_id, indexer, &mut state.index);
        }
        Metrics::record_kv_event_apply(worker_url, started.elapsed().as_secs_f64());
        Self::record_parentless(worker_url, &state.index.counters, &counters_before);
        if stored_blocks > 0 {
            Metrics::record_kv_event_blocks(worker_url, "stored", stored_blocks);
        }
        if removed_blocks > 0 {
            Metrics::record_kv_event_blocks(worker_url, "removed", removed_blocks);
        }
        Self::record_lag(worker_url, batch.timestamp);
        Metrics::record_kv_event_batch(worker_url, "applied");
        Metrics::set_kv_index_blocks(worker_url, indexer.worker_block_count(worker_id));
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
    fn admit_snapshot_chunk(
        batch: &KvEventBatch,
        chunk: &KvSnapshotChunk,
        worker_url: &str,
        worker_id: u32,
        indexer: &KvIndex,
        state: &mut WorkerStreamState,
        on_batch: &mut impl FnMut(&KvEventBatch),
    ) -> BatchOutcome {
        let rank = batch.dp_rank.unwrap_or(0);
        if chunk.index == 0 || state.snapshot.is_none() {
            if chunk.index != 0 {
                warn!(
                    worker_url = %worker_url,
                    rank,
                    index = chunk.index,
                    "KV event relay snapshot arrived without its first chunk; taking it as a resync"
                );
            }
            info!(
                worker_url = %worker_url,
                rank,
                chunks = chunk.count,
                blocks = chunk.blocks,
                unknown_before = chunk.unknown_before,
                through = batch.sequence_number + u64::from(chunk.count.saturating_sub(chunk.index + 1)),
                "KV event relay served a state snapshot; replacing the worker's index state"
            );
            if chunk.unknown_before > 0 {
                warn!(
                    worker_url = %worker_url,
                    rank,
                    unknown_before = chunk.unknown_before,
                    "KV event relay snapshot starts late: the engine's blocks from before the \
                     relay's record are unknown; the worker's index is partial until they leave \
                     the engine (rank degraded)"
                );
            }
            Self::apply_cleared(worker_id, indexer, &mut state.index);
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
        cursor.resync_to(batch.sequence_number);
        if chunk.unknown_before > 0 {
            cursor.mark_degraded();
        }
        on_batch(batch);
        let counters_before = state.index.counters;
        let mut snapshot_blocks = 0usize;
        for event in &batch.events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                snapshot_blocks += stored.blocks.len();
            }
            Self::apply_event(event, worker_id, indexer, &mut state.index);
        }
        Self::record_parentless(worker_url, &state.index.counters, &counters_before);
        if snapshot_blocks > 0 {
            Metrics::record_kv_event_blocks(worker_url, "snapshot", snapshot_blocks);
        }
        Self::record_lag(worker_url, batch.timestamp);
        Metrics::record_kv_event_batch(worker_url, "snapshot");
        Metrics::set_kv_index_blocks(worker_url, indexer.worker_block_count(worker_id));
        if let Some(progress) = &mut state.snapshot {
            progress.applied += 1;
            if progress.applied >= progress.count {
                info!(
                    worker_url = %worker_url,
                    chunks = progress.count,
                    blocks = progress.blocks,
                    through = batch.sequence_number,
                    "KV event relay snapshot applied; continuing with live events"
                );
                state.snapshot = None;
                Metrics::set_kv_event_degraded_ranks(worker_url, state.degraded_ranks());
            }
        }
        BatchOutcome::Applied
    }

    /// One rank's publisher lost its history (a restart, or an unreplayable
    /// gap too large to keep): the worker's pooled index state goes with it,
    /// and the other ranks' cursors start over so their next batch is taken
    /// as a first one instead of clearing the index a second time.
    fn clear_worker(worker_id: u32, indexer: &KvIndex, state: &mut WorkerStreamState, rank: i32) {
        Self::apply_cleared(worker_id, indexer, &mut state.index);
        for (other, cursor) in &mut state.ranks {
            if *other != rank {
                cursor.reset();
            }
        }
    }

    /// Publish the parent-less stores a batch produced, if any.
    fn record_parentless(
        worker_url: &str,
        after: &WorkerIndexCounters,
        before: &WorkerIndexCounters,
    ) {
        let (stores, blocks) = after.parentless_since(before);
        if stores > 0 {
            Metrics::record_kv_event_parentless(worker_url, stores, blocks);
        }
    }

    /// The server declared its history gone: drop the worker's index state
    /// and every cursor, so the next stream is taken from wherever it starts.
    pub(super) fn reset_worker(
        indexer: &KvIndex,
        worker_id: u32,
        state: &mut WorkerStreamState,
        worker_url: &str,
        reason: ResyncReason,
    ) {
        Self::apply_cleared(worker_id, indexer, &mut state.index);
        for cursor in state.ranks.values_mut() {
            cursor.reset();
        }
        state.snapshot = None;
        Metrics::record_kv_event_resync(worker_url, reason.as_str());
        Metrics::set_kv_event_degraded_ranks(worker_url, 0);
        Metrics::set_kv_index_blocks(worker_url, indexer.worker_block_count(worker_id));
    }

    /// Age of a batch when applied, from the publisher's wall-clock stamp.
    fn record_lag(worker_url: &str, published_at: f64) {
        if published_at <= 0.0 {
            return;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let lag = now - published_at;
        if lag.is_finite() && lag >= 0.0 {
            Metrics::record_kv_event_lag(worker_url, lag);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use kv_index::{
        salt::namespace_seed, ContentHash, ReferenceIndexer, SequenceHash, StoredBlock,
    };
    use smg_grpc_client::common_proto::{
        EngineLoad, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheCleared, KvCacheEvent,
        KvCacheTier,
    };

    use super::{super::apply::convert_kv_block, *};
    use crate::worker::{
        kv_event_recovery::{Cursor, RESTART_WINDOW},
        kv_index_backend::KvIndexKind,
    };

    /// Token ids for engine block `id`: distinct content per id.
    fn tokens_for(id: i64) -> Vec<u32> {
        (0..4u32).map(|i| (id as u32) * 16 + i).collect()
    }

    fn kv_block(id: i64) -> KvBlock {
        KvBlock {
            block_hash: id,
            token_ids: tokens_for(id),
            block_size: 4,
            ..Default::default()
        }
    }

    fn stored(parent: Option<i64>, ids: &[i64]) -> KvCacheEvent {
        KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: ids.iter().map(|&id| kv_block(id)).collect(),
                parent_block_hash: parent,
                ..Default::default()
            })),
        }
    }

    fn removed(ids: &[i64]) -> KvCacheEvent {
        KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
                block_hashes: ids.to_vec(),
                ..Default::default()
            })),
        }
    }

    fn cleared() -> KvCacheEvent {
        KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
        }
    }

    fn batch(seq: u64, rank: Option<i32>, events: Vec<KvCacheEvent>) -> KvEventBatch {
        KvEventBatch {
            sequence_number: seq,
            timestamp: 0.0,
            events,
            dp_rank: rank,
            snapshot: None,
            load: None,
        }
    }

    /// `smg_kv_index_blocks{worker}` is set where applied batches are counted,
    /// from the index's own per-worker counter: it follows stores, removals and
    /// a clear, and costs the lookup path nothing. Both indexes keep that
    /// counter, so the gauge reads the same under `--kv-index chain`.
    #[test]
    fn index_block_gauge_follows_stores_removals_and_a_clear() {
        for kind in [KvIndexKind::Positional, KvIndexKind::Chain] {
            index_block_gauge_follows_the_index(kind);
        }
    }

    fn index_block_gauge_follows_the_index(kind: KvIndexKind) {
        use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

        fn index_blocks(handle: &PrometheusHandle) -> Option<f64> {
            handle
                .render()
                .lines()
                .find(|line| line.starts_with("smg_kv_index_blocks{worker=\"grpc://w1:9000\"}"))
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        }

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let mut sim = Sim::with_kind(kind);
            assert_eq!(index_blocks(&handle), None, "nothing applied yet");

            assert_eq!(
                sim.feed(&batch(1, None, vec![stored(None, &[1, 2, 3])])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 3);
            assert_eq!(
                index_blocks(&handle),
                Some(3.0),
                "{kind:?}: three blocks stored"
            );

            assert_eq!(
                sim.feed(&batch(2, None, vec![removed(&[3])])),
                BatchOutcome::Applied
            );
            assert_eq!(index_blocks(&handle), Some(2.0), "{kind:?}: one removed");

            assert_eq!(
                sim.feed(&batch(3, None, vec![stored(Some(2), &[4, 5])])),
                BatchOutcome::Applied
            );
            assert_eq!(
                index_blocks(&handle),
                Some(4.0),
                "{kind:?}: two more stored"
            );

            assert_eq!(
                sim.feed(&batch(4, None, vec![cleared()])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 0);
            assert_eq!(index_blocks(&handle), Some(0.0), "{kind:?}: cleared");
        });
    }

    /// A subscription served by a relay snapshot accounts its blocks the way
    /// the live path does: once the chunks are applied the worker's gauge
    /// reads the index's count, the snapshot's blocks are counted under
    /// `op="snapshot"`, and the live events after it keep counting from there.
    #[test]
    fn a_relay_snapshot_sets_the_worker_gauge_and_counts_its_blocks() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let series = |name: &str, labels: &[&str]| -> Option<f64> {
            handle
                .render()
                .lines()
                .find(|line| {
                    line.starts_with(name) && labels.iter().all(|label| line.contains(label))
                })
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        };
        let worker = "worker=\"grpc://w1:9000\"";
        metrics::with_local_recorder(&recorder, || {
            let mut sim = Sim::new();
            // The relay's live set at the cut (sequence 5): five blocks in two chunks.
            let chunks = snapshot_chunks(
                5,
                None,
                &[
                    vec![stored(None, &[1, 2, 3])],
                    vec![stored(Some(3), &[4, 5])],
                ],
                5,
            );
            for chunk in &chunks {
                assert_eq!(sim.feed(chunk), BatchOutcome::Applied);
            }
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 5);
            assert_eq!(
                series("smg_kv_index_blocks{", &[worker]),
                Some(5.0),
                "the gauge reads the snapshot's live set"
            );
            assert_eq!(
                series("smg_kv_event_blocks_total{", &[worker, "op=\"snapshot\""]),
                Some(5.0),
                "the snapshot's blocks are counted"
            );
            assert_eq!(
                series("smg_kv_event_blocks_total{", &[worker, "op=\"stored\""]),
                None,
                "a snapshot is not a publisher's store"
            );

            // Live events after the cut keep both moving.
            assert_eq!(
                sim.feed(&batch(6, None, vec![stored(Some(5), &[6, 7])])),
                BatchOutcome::Applied
            );
            assert_eq!(
                sim.feed(&batch(7, None, vec![removed(&[7])])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 6);
            assert_eq!(series("smg_kv_index_blocks{", &[worker]), Some(6.0));
            assert_eq!(
                series("smg_kv_event_blocks_total{", &[worker, "op=\"stored\""]),
                Some(2.0)
            );
            assert_eq!(
                series("smg_kv_event_blocks_total{", &[worker, "op=\"removed\""]),
                Some(1.0)
            );
            assert_eq!(
                series("smg_kv_event_blocks_total{", &[worker, "op=\"snapshot\""]),
                Some(5.0),
                "the snapshot count stands"
            );
        });
    }

    /// The production subscriber's per-worker state next to a reference that
    /// sees the stream the subscriber *should* have applied.
    struct Sim {
        indexer: KvIndex,
        worker: u32,
        state: WorkerStreamState,
        reference: ReferenceIndexer,
    }

    impl Sim {
        fn new() -> Self {
            Self::with_kind(KvIndexKind::Positional)
        }

        /// The same harness over the index selected by `--kv-index`.
        fn with_kind(kind: KvIndexKind) -> Self {
            let indexer = KvIndex::new(kind, 8);
            let worker = indexer.intern_worker("grpc://w1:9000").unwrap();
            Self {
                indexer,
                worker,
                state: WorkerStreamState::default(),
                reference: ReferenceIndexer::new(),
            }
        }

        /// Feed a batch through the real admission path.
        fn feed(&mut self, b: &KvEventBatch) -> BatchOutcome {
            KvEventMonitor::admit_batch(
                b,
                "grpc://w1:9000",
                self.worker,
                &self.indexer,
                &mut self.state,
                &mut |_: &KvEventBatch| {},
            )
        }

        /// Apply a batch to the reference with the subscriber's semantics
        /// (a store whose parent is unknown starts a new chain).
        fn reference_apply(&mut self, b: &KvEventBatch) {
            let seed = namespace_seed(None, None);
            for event in &b.events {
                match event.data.as_ref().unwrap() {
                    kv_cache_event::Data::Stored(st) => {
                        let blocks: Vec<StoredBlock> = st
                            .blocks
                            .iter()
                            .map(|block| convert_kv_block(block, seed))
                            .collect();
                        let parent = st.parent_block_hash.map(SequenceHash::from);
                        if self
                            .reference
                            .apply_stored(self.worker, &blocks, parent)
                            .is_err()
                        {
                            self.reference
                                .apply_stored(self.worker, &blocks, None)
                                .unwrap();
                        }
                    }
                    kv_cache_event::Data::Removed(rm) => {
                        let hashes: Vec<SequenceHash> = rm
                            .block_hashes
                            .iter()
                            .map(|&h| SequenceHash::from(h))
                            .collect();
                        self.reference.apply_removed(self.worker, &hashes);
                    }
                    kv_cache_event::Data::Cleared(_) => self.reference.apply_cleared(self.worker),
                }
            }
        }

        /// Feed to both: what a correctly delivered batch does.
        fn deliver(&mut self, b: &KvEventBatch) -> BatchOutcome {
            self.reference_apply(b);
            self.feed(b)
        }

        /// Apply a batch straight into the worker's index state, bypassing
        /// admission (how a snapshot or a buffered tail lands).
        fn apply_direct(&mut self, b: &KvEventBatch) {
            for event in &b.events {
                KvEventMonitor::apply_event(
                    event,
                    self.worker,
                    &self.indexer,
                    &mut self.state.index,
                );
            }
        }

        fn assert_matches_reference(&self) {
            let production: BTreeSet<(u32, usize, ContentHash, SequenceHash)> =
                self.indexer.debug_blocks().into_iter().collect();
            assert_eq!(production, self.reference.blocks(), "index content");
            for query in self.queries() {
                let scores = self.indexer.find_matches(&query, false).scores;
                let expected = self.reference.find_matches(&query);
                let got: Vec<(u32, u32)> = scores.into_iter().collect();
                let want: Vec<(u32, u32)> = expected.into_iter().filter(|(_, s)| *s > 0).collect();
                assert_eq!(got, want, "scores for {query:?}");
            }
        }

        /// Lookups: every stored chain, plus a mutated copy of each.
        fn queries(&self) -> Vec<Vec<ContentHash>> {
            let mut chains: Vec<Vec<ContentHash>> = Vec::new();
            let blocks = self.reference.blocks();
            let max_pos = blocks.iter().map(|b| b.1).max().unwrap_or(0);
            // Rebuild chains by walking positions from the reference content.
            let mut by_pos: Vec<Vec<ContentHash>> = vec![Vec::new(); max_pos + 1];
            for (_, pos, content, _) in &blocks {
                if !by_pos[*pos].contains(content) {
                    by_pos[*pos].push(*content);
                }
            }
            let mut chain = Vec::new();
            for level in &by_pos {
                if let Some(c) = level.first() {
                    chain.push(*c);
                    chains.push(chain.clone());
                }
            }
            if let Some(full) = chains.last().cloned() {
                let mut mutated = full.clone();
                if mutated.len() > 1 {
                    mutated[1] = ContentHash(0xDEAD_BEEF);
                    chains.push(mutated);
                }
            }
            chains.push(vec![ContentHash(1), ContentHash(2)]);
            chains
        }
    }

    #[test]
    fn recovery_in_order_stream_matches_reference() {
        let mut sim = Sim::new();
        assert_eq!(
            sim.deliver(&batch(1, None, vec![stored(None, &[1, 2, 3])])),
            BatchOutcome::Applied
        );
        assert_eq!(
            sim.deliver(&batch(2, None, vec![stored(Some(3), &[4, 5])])),
            BatchOutcome::Applied
        );
        assert_eq!(
            sim.deliver(&batch(3, None, vec![removed(&[5])])),
            BatchOutcome::Applied
        );
        assert_eq!(
            sim.deliver(&batch(4, None, vec![stored(Some(4), &[6])])),
            BatchOutcome::Applied
        );
        sim.assert_matches_reference();
    }

    #[test]
    fn recovery_stream_numbered_from_zero_matches_reference() {
        // vLLM and SGLang publishers count from 0; 0 must not be treated as
        // "no cursor" once a batch carried it.
        let mut sim = Sim::new();
        assert_eq!(
            sim.deliver(&batch(0, None, vec![cleared(), stored(None, &[1, 2])])),
            BatchOutcome::Applied
        );
        assert_eq!(
            sim.deliver(&batch(1, None, vec![stored(Some(2), &[3])])),
            BatchOutcome::Applied
        );
        assert_eq!(
            sim.feed(&batch(1, None, vec![removed(&[1])])),
            BatchOutcome::Skipped
        );
        assert_eq!(sim.state.resume_sequence(), 1);
        sim.assert_matches_reference();
    }

    #[test]
    fn recovery_gap_filled_by_replay_matches_reference() {
        let mut sim = Sim::new();
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
        sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
        // Batch 3 is lost on the wire; 4 arrives: one replay is asked for.
        let b3 = batch(3, None, vec![removed(&[3])]);
        let b4 = batch(4, None, vec![stored(Some(2), &[7])]);
        assert_eq!(
            sim.feed(&b4),
            BatchOutcome::Gap {
                expected: 3,
                received: 4
            }
        );
        assert_eq!(sim.state.resume_sequence(), 2);
        // Resuming after 2, the server replays 3 and continues live.
        assert_eq!(sim.deliver(&b3), BatchOutcome::Applied);
        assert_eq!(sim.deliver(&b4), BatchOutcome::Applied);
        assert_eq!(
            sim.deliver(&batch(5, None, vec![stored(Some(7), &[8])])),
            BatchOutcome::Applied
        );
        assert!(!sim.state.ranks[&0].is_degraded());
        sim.assert_matches_reference();
    }

    #[test]
    fn recovery_gap_without_replay_keeps_state_and_continues() {
        let mut sim = Sim::new();
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
        sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
        // Batches 3 and 4 are lost; the server has no history (the Rust
        // relay): after the replay request it streams 5 again.
        let b5 = batch(5, None, vec![stored(Some(3), &[9])]);
        assert_eq!(
            sim.feed(&b5),
            BatchOutcome::Gap {
                expected: 3,
                received: 5
            }
        );
        assert_eq!(sim.deliver(&b5), BatchOutcome::Applied);
        assert!(sim.state.ranks[&0].is_degraded());
        // Everything after keeps applying; no reconnect loop.
        assert_eq!(
            sim.deliver(&batch(6, None, vec![stored(Some(9), &[10])])),
            BatchOutcome::Applied
        );
        // The reference saw the same stream minus the lost batches.
        sim.assert_matches_reference();
        // A later gap starts a fresh single replay attempt.
        assert_eq!(
            sim.feed(&batch(8, None, vec![])),
            BatchOutcome::Gap {
                expected: 7,
                received: 8
            }
        );
    }

    #[test]
    fn recovery_duplicates_and_out_of_order_batches_are_skipped() {
        let mut sim = Sim::new();
        let b1 = batch(1, None, vec![stored(None, &[1, 2])]);
        let b2 = batch(2, None, vec![stored(Some(2), &[3])]);
        let b3 = batch(3, None, vec![removed(&[3])]);
        let b4 = batch(4, None, vec![stored(Some(2), &[4])]);
        sim.deliver(&b1);
        sim.deliver(&b2);
        sim.deliver(&b3);
        // Replayed overlap after a reconnect: already applied, must not re-apply.
        // (A duplicate of sequence 1 would be taken as a new publisher: see
        // `a_counter_at_its_start_below_the_cursor_is_a_restart`.)
        assert_eq!(sim.feed(&b2), BatchOutcome::Skipped);
        assert_eq!(sim.feed(&b3), BatchOutcome::Skipped);
        sim.deliver(&b4);
        sim.assert_matches_reference();
    }

    #[test]
    fn recovery_publisher_restart_clears_the_rank() {
        let mut sim = Sim::new();
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2, 3])]));
        for seq in 2..=(RESTART_WINDOW + 5) {
            sim.deliver(&batch(seq, None, vec![]));
        }
        // The engine restarts: its cache is empty and it counts from 0 again.
        let fresh = batch(0, None, vec![stored(None, &[21, 22])]);
        sim.reference.apply_cleared(sim.worker);
        sim.reference_apply(&fresh);
        assert_eq!(sim.feed(&fresh), BatchOutcome::Applied);
        assert_eq!(
            sim.deliver(&batch(1, None, vec![stored(Some(22), &[23])])),
            BatchOutcome::Applied
        );
        assert!(!sim.state.ranks[&0].is_degraded());
        sim.assert_matches_reference();
    }

    #[test]
    fn recovery_restart_seen_on_a_fresh_connection_clears_the_rank() {
        let mut sim = Sim::new();
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
        sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
        // The worker died and came back: the stream reconnected and the new
        // publisher counts from 1 with an empty cache (the relay and the mock
        // engine stream live; neither can replay).
        sim.state.ranks.get_mut(&0).unwrap().reconnected();
        let fresh = batch(1, None, vec![stored(None, &[7])]);
        sim.reference.apply_cleared(sim.worker);
        sim.reference_apply(&fresh);
        assert_eq!(sim.feed(&fresh), BatchOutcome::Applied);
        sim.assert_matches_reference();
        assert_eq!(sim.state.resume_sequence(), 1);
    }

    #[test]
    fn recovery_clear_below_the_cursor_is_a_restart() {
        let mut sim = Sim::new();
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
        sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
        // SGLang's first batch after a restart carries AllBlocksCleared and
        // its counter starts over, with the servicer's stream still up.
        let first = batch(0, None, vec![cleared(), stored(None, &[9])]);
        sim.reference.apply_cleared(sim.worker);
        sim.reference_apply(&first);
        assert_eq!(sim.feed(&first), BatchOutcome::Applied);
        sim.assert_matches_reference();
        // A residency agent's clear below the cursor is just a duplicate.
        let agent = KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Cleared(KvCacheCleared {
                ownership: Some("kvcr".to_string()),
            })),
        };
        assert_eq!(
            sim.feed(&batch(0, None, vec![agent])),
            BatchOutcome::Skipped
        );
        sim.assert_matches_reference();
    }

    #[test]
    fn recovery_cleared_event_matches_reference() {
        let mut sim = Sim::new();
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2, 3])]));
        sim.deliver(&batch(2, None, vec![cleared()]));
        sim.deliver(&batch(3, None, vec![stored(None, &[4])]));
        sim.assert_matches_reference();
        assert_eq!(sim.reference.worker_block_count(sim.worker), 1);
    }

    #[test]
    fn recovery_dp_ranks_keep_independent_cursors() {
        let mut sim = Sim::new();
        // Two publishers, each numbering from 1, interleaved on one stream.
        sim.deliver(&batch(1, Some(0), vec![stored(None, &[1, 2])]));
        sim.deliver(&batch(1, Some(1), vec![stored(None, &[101, 102])]));
        sim.deliver(&batch(2, Some(1), vec![stored(Some(102), &[103])]));
        sim.deliver(&batch(2, Some(0), vec![stored(Some(2), &[3])]));
        assert_eq!(
            sim.deliver(&batch(3, Some(0), vec![removed(&[3])])),
            BatchOutcome::Applied
        );
        // Each rank dedups against its own cursor.
        assert_eq!(
            sim.feed(&batch(2, Some(1), vec![stored(None, &[200])])),
            BatchOutcome::Skipped
        );
        assert_eq!(
            sim.deliver(&batch(3, Some(1), vec![stored(Some(103), &[104])])),
            BatchOutcome::Applied
        );
        assert_eq!(sim.state.ranks.len(), 2);
        assert_eq!(sim.state.degraded_ranks(), 0);
        assert_eq!(sim.state.resume_sequence(), 3);
        sim.assert_matches_reference();
        // Rank 1's publisher restarts: the worker's pooled index state is
        // cleared and rank 0's cursor starts over, so its next batch is taken
        // as a first one rather than clearing the index again.
        sim.state.reconnected();
        let fresh = batch(1, Some(1), vec![stored(None, &[111])]);
        sim.reference.apply_cleared(sim.worker);
        sim.reference_apply(&fresh);
        assert_eq!(sim.feed(&fresh), BatchOutcome::Applied);
        assert_eq!(
            sim.deliver(&batch(4, Some(0), vec![stored(None, &[5])])),
            BatchOutcome::Applied
        );
        sim.assert_matches_reference();
        assert_eq!(sim.state.resume_sequence(), 4);
    }

    #[test]
    fn recovery_resync_forgets_copy_counts() {
        // Copy counts live in the pooled index state and cursors in the rank
        // state; a restart clears the counts with the blocks, so no stale
        // host copy keeps a block routable afterwards.
        let mut sim = Sim::new();
        let mut on_host = stored(None, &[1, 2]);
        if let Some(kv_cache_event::Data::Stored(st)) = on_host.data.as_mut() {
            st.tier = Some(KvCacheTier::Host as i32);
        }
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
        sim.deliver(&batch(2, None, vec![on_host]));
        sim.state.reconnected();
        let fresh = batch(1, None, vec![stored(None, &[1, 2])]);
        sim.reference.apply_cleared(sim.worker);
        sim.reference_apply(&fresh);
        assert_eq!(sim.feed(&fresh), BatchOutcome::Applied);
        assert_eq!(
            sim.deliver(&batch(2, None, vec![removed(&[2])])),
            BatchOutcome::Applied
        );
        sim.assert_matches_reference();
        assert_eq!(sim.reference.worker_block_count(sim.worker), 1);
    }

    /// The chunks a relay sends for a live set, stamped up to `through`:
    /// chunk 0 begins with the clear, every chunk is marked.
    fn snapshot_chunks(
        through: u64,
        rank: Option<i32>,
        stores_per_chunk: &[Vec<KvCacheEvent>],
        blocks: u64,
    ) -> Vec<KvEventBatch> {
        let count = stores_per_chunk.len() as u32;
        stores_per_chunk
            .iter()
            .enumerate()
            .map(|(index, stores)| {
                let mut events = Vec::new();
                if index == 0 {
                    events.push(cleared());
                }
                events.extend(stores.iter().cloned());
                KvEventBatch {
                    sequence_number: through + 1 - u64::from(count) + index as u64,
                    timestamp: 0.0,
                    events,
                    dp_rank: rank,
                    snapshot: Some(KvSnapshotChunk {
                        index: index as u32,
                        count,
                        blocks,
                        unknown_before: 0,
                    }),
                    load: None,
                }
            })
            .collect()
    }

    fn index_blocks(sim: &Sim) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        sim.indexer.debug_blocks().into_iter().collect()
    }

    /// A gateway that starts after the relay's history rolled receives the
    /// live set as a snapshot: applied as one `snapshot` resync, it leaves
    /// the index equal to the one that saw the whole stream (and to the
    /// reference), and the live stream continues from the cut with nothing
    /// skipped or repeated.
    #[test]
    fn a_relay_snapshot_is_applied_as_a_resync_and_rebuilds_the_live_set() {
        let mut seen = Sim::new();
        seen.deliver(&batch(1, None, vec![stored(None, &[1, 2, 3])]));
        seen.deliver(&batch(2, None, vec![stored(Some(3), &[4, 5])]));
        seen.deliver(&batch(3, None, vec![stored(None, &[10])]));
        seen.deliver(&batch(4, None, vec![removed(&[5, 10])]));
        seen.deliver(&batch(5, None, vec![stored(Some(2), &[6])]));
        seen.assert_matches_reference();

        // Live set at the cut (sequence 5): 1, 2, 3, 4 and the branch 6.
        let chunks = snapshot_chunks(
            5,
            None,
            &[
                vec![stored(None, &[1, 2, 3])],
                vec![stored(Some(3), &[4]), stored(Some(2), &[6])],
            ],
            5,
        );
        let mut fresh = Sim::new();
        for chunk in &chunks {
            assert_eq!(fresh.feed(chunk), BatchOutcome::Applied);
        }
        assert!(fresh.state.snapshot.is_none(), "both chunks arrived");
        assert_eq!(fresh.state.ranks[&0].cursor(), Cursor::Live(5));
        assert_eq!(fresh.state.resume_sequence(), 5);
        assert_eq!(
            index_blocks(&fresh),
            index_blocks(&seen),
            "index after the snapshot"
        );
        assert_eq!(
            index_blocks(&fresh),
            seen.reference.blocks(),
            "against the reference"
        );

        // The cut's own sequence is behind the cursor; the next one applies.
        assert_eq!(
            fresh.feed(&batch(5, None, vec![stored(None, &[99])])),
            BatchOutcome::Skipped
        );
        let live = batch(6, None, vec![stored(Some(6), &[7]), removed(&[4])]);
        assert_eq!(fresh.feed(&live), BatchOutcome::Applied);
        assert_eq!(seen.deliver(&live), BatchOutcome::Applied);
        seen.assert_matches_reference();
        assert_eq!(
            index_blocks(&fresh),
            index_blocks(&seen),
            "index after the live batch"
        );
        assert_eq!(seen.reference.worker_block_count(seen.worker), 5);
    }

    /// The snapshot lists a block once per physical copy the engine still
    /// holds, so the copy counts after it match a gateway that saw every
    /// store and removal: the same removals evict the same blocks.
    #[test]
    fn a_relay_snapshot_carries_the_copies_the_engine_still_holds() {
        let mut seen = Sim::new();
        seen.feed(&batch(1, None, vec![stored(None, &[1, 2])]));
        // A second copy of 1 and of 2, one copy of 1 removed since.
        seen.feed(&batch(
            2,
            None,
            vec![stored(None, &[1]), stored(Some(1), &[2])],
        ));
        seen.feed(&batch(3, None, vec![removed(&[1])]));
        let chunks = snapshot_chunks(
            3,
            None,
            &[vec![stored(None, &[1, 2]), stored(Some(1), &[2])]],
            3,
        );
        let mut fresh = Sim::new();
        assert_eq!(fresh.feed(&chunks[0]), BatchOutcome::Applied);
        assert_eq!(index_blocks(&fresh), index_blocks(&seen));
        for (seq, hashes) in [(4, &[1][..]), (5, &[2][..]), (6, &[2][..])] {
            let removal = batch(seq, None, vec![removed(hashes)]);
            assert_eq!(fresh.feed(&removal), BatchOutcome::Applied);
            assert_eq!(seen.feed(&removal), BatchOutcome::Applied);
            assert_eq!(index_blocks(&fresh), index_blocks(&seen), "after {seq}");
        }
        assert!(index_blocks(&fresh).is_empty(), "every copy is gone");
    }

    /// A snapshot chunk is a resync wherever the rank's cursor stands: the
    /// gap and duplicate rules do not apply to it, the worker's old blocks
    /// go, the cursor moves to the chunk's stamp.
    #[test]
    fn a_snapshot_replaces_whatever_the_rank_held_and_moves_its_cursor() {
        let mut sim = Sim::new();
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
        sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
        assert_eq!(
            sim.feed(&batch(9, None, vec![stored(None, &[9])])),
            BatchOutcome::Gap {
                expected: 3,
                received: 9
            }
        );
        // The relay's answer to the resubscription: its window had rolled.
        let chunks = snapshot_chunks(40, None, &[vec![stored(None, &[7, 8])]], 2);
        assert_eq!(sim.feed(&chunks[0]), BatchOutcome::Applied);
        sim.reference_apply(&chunks[0]);
        sim.assert_matches_reference();
        assert_eq!(sim.state.ranks[&0].cursor(), Cursor::Live(40));
        assert!(sim.state.ranks[&0].replay_pending().is_none());
        assert!(!sim.state.ranks[&0].is_degraded());
        assert_eq!(sim.reference.worker_block_count(sim.worker), 2);
        let next = batch(41, None, vec![removed(&[7])]);
        assert_eq!(sim.deliver(&next), BatchOutcome::Applied);
        sim.assert_matches_reference();
        assert_eq!(sim.reference.worker_block_count(sim.worker), 1);
    }

    /// A batch that carries only a load record repeats the last sequence and
    /// has no events: it is recognised before admission, so it is neither a
    /// duplicate nor a restart to the rank's cursor.
    #[test]
    fn a_load_only_batch_is_recognised_before_admission() {
        let record = EngineLoad {
            running_requests: 3,
            load_only: true,
            ..Default::default()
        };
        let mut only = batch(7, None, vec![]);
        only.load = Some(record.clone());
        assert!(KvEventMonitor::is_load_only(&only));
        let mut carrying = batch(8, None, vec![stored(None, &[1])]);
        carrying.load = Some(EngineLoad {
            load_only: false,
            ..record
        });
        assert!(!KvEventMonitor::is_load_only(&carrying));
        assert!(!KvEventMonitor::is_load_only(&batch(9, None, vec![])));
    }

    /// A snapshot whose relay joined the publisher late and could not replay
    /// the start leaves the rank degraded, as an unrecovered gap would; a
    /// whole one lifts it.
    #[test]
    fn a_snapshot_that_starts_late_marks_the_rank_degraded() {
        let mut sim = Sim::new();
        let mut late = snapshot_chunks(9, None, &[vec![stored(None, &[1])]], 1);
        late[0].snapshot.as_mut().unwrap().unknown_before = 40;
        assert_eq!(sim.feed(&late[0]), BatchOutcome::Applied);
        assert!(sim.state.ranks[&0].is_degraded());
        assert_eq!(sim.state.degraded_ranks(), 1);
        assert_eq!(
            sim.feed(&batch(10, None, vec![stored(Some(1), &[2])])),
            BatchOutcome::Applied
        );
        assert!(sim.state.ranks[&0].is_degraded(), "until the next resync");
        let whole = snapshot_chunks(12, None, &[vec![stored(None, &[1])]], 1);
        assert_eq!(sim.feed(&whole[0]), BatchOutcome::Applied);
        assert!(!sim.state.ranks[&0].is_degraded());
        assert_eq!(sim.state.degraded_ranks(), 0);
    }

    /// A stream that ends with chunks still owed leaves a partial live set:
    /// the cursors are forgotten so the next subscription asks from zero and
    /// gets a whole snapshot; a complete one leaves nothing to abandon.
    #[test]
    fn a_stream_that_ends_mid_snapshot_starts_the_next_subscription_from_zero() {
        let mut sim = Sim::new();
        let chunks = snapshot_chunks(
            10,
            None,
            &[vec![stored(None, &[1])], vec![stored(Some(1), &[2])]],
            2,
        );
        assert_eq!(sim.feed(&chunks[0]), BatchOutcome::Applied);
        assert_eq!(
            sim.state.snapshot,
            Some(SnapshotProgress {
                count: 2,
                applied: 1,
                blocks: 2
            })
        );
        assert_eq!(sim.state.resume_sequence(), 9);
        assert!(sim.state.abandon_snapshot());
        assert_eq!(sim.state.resume_sequence(), 0);
        assert!(!sim.state.abandon_snapshot());
        for chunk in &chunks {
            assert_eq!(sim.feed(chunk), BatchOutcome::Applied);
        }
        assert!(sim.state.snapshot.is_none());
        assert_eq!(sim.state.resume_sequence(), 10);
        assert!(!sim.state.abandon_snapshot());
    }

    #[test]
    fn recovery_snapshot_resync_applies_the_tail_in_order() {
        let mut sim = Sim::new();
        sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
        sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
        // Lost history: the subscriber asks for a snapshot out of band and
        // holds live batches meanwhile.
        sim.state.ranks.get_mut(&0).unwrap().begin_snapshot();
        let live3 = batch(3, None, vec![removed(&[3])]);
        let live4 = batch(4, None, vec![stored(Some(2), &[5])]);
        assert_eq!(sim.feed(&live3), BatchOutcome::Skipped);
        assert_eq!(sim.feed(&live4), BatchOutcome::Skipped);
        assert_eq!(sim.state.ranks[&0].tail_len(), 2);
        // The snapshot (engine state through sequence 2) replaces the rank.
        let snapshot = batch(
            2,
            None,
            vec![cleared(), stored(None, &[1, 2]), stored(Some(2), &[3])],
        );
        KvEventMonitor::apply_cleared(sim.worker, &sim.indexer, &mut sim.state.index);
        sim.apply_direct(&snapshot);
        let tail = sim
            .state
            .ranks
            .get_mut(&0)
            .unwrap()
            .finish_snapshot(2)
            .expect("tail intact");
        assert_eq!(tail.len(), 2);
        for b in &tail {
            sim.apply_direct(b);
        }
        sim.reference_apply(&live3);
        sim.reference_apply(&live4);
        assert_eq!(
            sim.deliver(&batch(5, None, vec![stored(Some(5), &[6])])),
            BatchOutcome::Applied
        );
        assert!(!sim.state.ranks[&0].is_degraded());
        sim.assert_matches_reference();
    }

    /// A store whose parent the index does not hold is placed from the root
    /// and counted, per worker, so a soak can see how much of a chain's
    /// content arrives detached (where the chain index fragments first).
    #[test]
    fn parentless_stores_are_counted_per_worker() {
        use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

        fn counter(handle: &PrometheusHandle, name: &str) -> Option<f64> {
            handle
                .render()
                .lines()
                .find(|line| line.starts_with(&format!("{name}{{worker=\"grpc://w1:9000\"}}")))
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        }

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let mut sim = Sim::new();
            sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
            assert_eq!(
                counter(&handle, "smg_kv_event_parentless_stores_total"),
                None
            );
            // The parent of this store was never seen (dropped or evicted).
            sim.deliver(&batch(2, None, vec![stored(Some(99), &[3, 4, 5])]));
            assert_eq!(
                counter(&handle, "smg_kv_event_parentless_stores_total"),
                Some(1.0)
            );
            assert_eq!(
                counter(&handle, "smg_kv_event_parentless_blocks_total"),
                Some(3.0)
            );
            assert_eq!(sim.state.index.counters.parentless_stores, 1);
            // A chained store after a held parent is not counted.
            sim.deliver(&batch(3, None, vec![stored(Some(2), &[6])]));
            assert_eq!(
                counter(&handle, "smg_kv_event_parentless_stores_total"),
                Some(1.0)
            );
            sim.assert_matches_reference();
        });
    }
}
