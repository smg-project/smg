//! One engine's stream state: the admission cursor per data-parallel rank,
//! gap recovery, relay snapshots, the rank workers the stream feeds, and the
//! metrics around an applied batch.

use std::{
    collections::HashMap,
    sync::Arc,
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
        endpoint::Endpoint,
        kv_event_recovery::{Admission, RankState, ResyncReason},
        kv_index_backend::KvIndex,
        Worker,
    },
};

/// A worker on its engine's stream: the rank it stands for, its id in the
/// model's index (an events subscription; a loads-only one has none) and its
/// own index state, whose copies are pooled across the ranks it indexes (see
/// [`WorkerIndexState`]). Cursors are the engine's, not the member's.
pub(super) struct Member {
    pub(super) worker: Arc<dyn Worker>,
    pub(super) url: String,
    /// The data-parallel rank this worker stands for, `None` for a worker
    /// that stands for its whole engine. The engine's stream names every
    /// rank's batches: a batch enters this member's index when it is its
    /// rank's, or unranked (every rank's).
    pub(super) rank: Option<i32>,
    pub(super) worker_id: Option<u32>,
    pub(super) index: WorkerIndexState,
}

impl Member {
    /// The member `worker` is on its engine's stream, interned in the
    /// model's index as `worker_id` (`None` on a loads-only stream, or when
    /// the index ran out of ids).
    pub(super) fn new(worker: Arc<dyn Worker>, worker_id: Option<u32>) -> Self {
        Self {
            url: worker.url().to_string(),
            rank: worker.dp_rank().and_then(|rank| i32::try_from(rank).ok()),
            worker,
            worker_id,
            index: WorkerIndexState::default(),
        }
    }

    /// Whether a batch of `rank` (`None`: unranked, every rank's) is this
    /// member's cache. Another rank's is counted, not indexed.
    pub(super) fn takes(&self, rank: Option<i32>) -> bool {
        match (self.rank, rank) {
            (Some(mine), Some(theirs)) => mine == theirs,
            _ => true,
        }
    }

    /// Whether a load record of `rank` (an unranked one is rank 0's) is this
    /// member's: its own rank's, or every rank's for a whole-engine member.
    pub(super) fn takes_record(&self, rank: i32) -> bool {
        self.rank.is_none_or(|mine| mine == rank)
    }

    /// Whether `rank`'s cursor carries batches this member indexes: every
    /// rank for a whole-engine member; its own rank for a pinned one, plus
    /// rank 0's once a batch that names no rank (every rank's, indexed here)
    /// has ridden it (`saw_unranked`).
    fn counts_rank(&self, rank: i32, saw_unranked: bool) -> bool {
        match self.rank {
            None => true,
            Some(mine) => mine == rank || (rank == 0 && saw_unranked),
        }
    }
}

/// One engine's subscription state: an admission cursor per data-parallel
/// rank (every publisher numbers its own batches), advanced once per engine,
/// the relay snapshot's progress, and the members the stream feeds, each with
/// its own index state. Cursors and copy counts never mix.
#[derive(Default)]
pub(super) struct EngineStreamState {
    /// The engine's address, for the log lines about the stream itself.
    pub(super) address: String,
    pub(super) ranks: HashMap<i32, RankState>,
    /// A relay snapshot whose chunks are still arriving on the stream.
    pub(super) snapshot: Option<SnapshotProgress>,
    /// Whether a batch that names no rank has ridden rank 0's cursor: such a
    /// batch is every rank's and is indexed by every member, so that cursor
    /// then counts toward a pinned member's degraded ranks too.
    pub(super) saw_unranked: bool,
    /// The rank workers on the stream, in the order they joined.
    pub(super) members: Vec<Member>,
    /// A member joined after the stream had delivered batches: the next
    /// round starts the engine over from zero, so the relay's history or
    /// snapshot reaches the member's rank too.
    pub(super) rejoin: bool,
    /// The ranks of the members that joined late since the last round
    /// (`None`: a member standing for the whole engine).
    pub(super) joined_late: Vec<Option<i32>>,
    /// The late-joined ranks whose history the stream has not delivered
    /// yet (no cursor so far): the next round starts from zero while any
    /// is listed, a failed attempt included; a rank leaves the list with
    /// its first batch or its last member, and the whole list goes once a
    /// round from zero reached live for an incumbent (a replay skip ended
    /// by a batch past its cursor, not by a gap sighting) or a relay
    /// snapshot came in whole: the joiner's window, if any is still to
    /// come, arrives on that stream or, after a break, through the refused
    /// cursor's restart from zero (a multi-rank relay refuses a nonzero
    /// cursor). An engine with no batch past any cursor (idle since the
    /// join) keeps asking for zero: a known limit, closed only by an
    /// end-of-history mark on the stream.
    pub(super) history_owed: Vec<i32>,
}

/// Where an in-band relay snapshot stands (`KvSnapshotChunk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SnapshotProgress {
    count: u32,
    applied: u32,
    blocks: u64,
}

impl EngineStreamState {
    /// The state of the engine at `address` with `members` on its stream.
    pub(super) fn new(address: &str, members: Vec<Member>) -> Self {
        Self {
            address: address.to_string(),
            members,
            ..Self::default()
        }
    }

    /// The cursor to send when resubscribing: rank 0's last applied sequence
    /// (the servicers replay rank 0's publisher). Other ranks keep their own
    /// cursors and dedup what arrives.
    pub(super) fn resume_sequence(&self) -> u64 {
        self.ranks.get(&0).map_or(0, RankState::resume_from)
    }

    /// Whether the next round asks for the engine's history from zero:
    /// while a late-joined rank still owes its history (no cursor yet, a
    /// member still standing for it), a failed attempt included, and until
    /// a round from zero reached live for an incumbent or a snapshot came
    /// in whole (`history_owed`). The ranks with cursors then skip the
    /// replay up to them (`RankState::replay_from_zero`); a round that
    /// resumes from the cursor ends that skip, so a sequence below the
    /// cursor reads as a publisher restart again.
    pub(super) fn starts_from_zero(&mut self) -> bool {
        let (ranks, members) = (&self.ranks, &self.members);
        self.history_owed.retain(|rank| {
            !ranks.contains_key(rank) && members.iter().any(|m| m.rank == Some(*rank))
        });
        let from_zero = !self.history_owed.is_empty();
        for cursor in self.ranks.values_mut() {
            if from_zero {
                cursor.replay_from_zero();
            } else {
                cursor.resume_from_cursor();
            }
        }
        from_zero
    }

    /// A new stream connection was made: every rank's next batch is its
    /// first on it.
    pub(super) fn reconnected(&mut self) {
        for rank in self.ranks.values_mut() {
            rank.reconnected();
        }
    }

    /// The member at `url` leaves the stream, with its index state for the
    /// cleanup; `None` when no such member is on it.
    pub(super) fn leave(&mut self, url: &str) -> Option<Member> {
        let at = self.members.iter().position(|member| member.url == url)?;
        Some(self.members.remove(at))
    }

    /// Whether the stream has delivered a batch: a member joining now has
    /// missed its rank's history.
    pub(super) fn streamed(&self) -> bool {
        !self.ranks.is_empty()
    }

    /// Whether some member indexes a batch of `rank`.
    fn taken(&self, rank: Option<i32>) -> bool {
        self.members.iter().any(|member| member.takes(rank))
    }

    /// The degraded cursors among the ranks `member` indexes: every rank of
    /// a whole-engine member; a pinned member's own rank, plus rank 0's
    /// cursor once a batch that names no rank (every rank's, indexed there)
    /// has ridden it. Another rank's unreplayed gap says nothing about this
    /// member's index.
    pub(super) fn degraded_ranks_of(&self, member: &Member) -> usize {
        self.ranks
            .iter()
            .filter(|(rank, cursor)| {
                member.counts_rank(**rank, self.saw_unranked) && cursor.is_degraded()
            })
            .count()
    }

    /// Republish every member's degraded-ranks gauge: a cursor's state
    /// changed, and which members it counts toward is theirs to say.
    fn publish_degraded(&self) {
        for member in &self.members {
            Metrics::set_kv_event_degraded_ranks(&member.url, self.degraded_ranks_of(member));
        }
    }

    /// Count a batch of `rank` under the members whose cache it is, or under
    /// every member when it is no member's (a rank no worker stands for).
    fn count_batch(&self, rank: Option<i32>, disposition: &'static str) {
        let taken = self.taken(rank);
        for member in self.members.iter().filter(|m| !taken || m.takes(rank)) {
            Metrics::record_kv_event_batch(&member.url, disposition);
        }
    }

    /// Count a gap on `rank`'s cursor the same way.
    fn count_gap(&self, rank: Option<i32>, outcome: &'static str, missed: u64) {
        let taken = self.taken(rank);
        for member in self.members.iter().filter(|m| !taken || m.takes(rank)) {
            Metrics::record_kv_event_gap(&member.url, outcome, missed);
        }
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
    /// Dropped (a duplicate), held (a snapshot tail) or another rank's
    /// (counted, not indexed).
    Skipped,
    /// A gap: the caller reconnects asking for a replay from `expected`.
    Gap { expected: u64, received: u64 },
}

impl KvEventMonitor {
    /// Run one batch through its rank's admission cursor and apply it to the
    /// index state of every member whose cache it is.
    pub(super) fn admit_batch(
        batch: &KvEventBatch,
        indexer: &KvIndex,
        state: &mut EngineStreamState,
        on_batch: &mut impl FnMut(&KvEventBatch),
    ) -> BatchOutcome {
        if let Some(chunk) = &batch.snapshot {
            return Self::admit_snapshot_chunk(batch, chunk, indexer, state, on_batch);
        }
        let rank = batch.dp_rank.unwrap_or(0);
        // The first batch that names no rank puts rank 0's cursor into a
        // pinned member's degraded count: a change of the gauge in its own
        // right, republished below with the rest.
        let mut degraded_changed = false;
        if batch.dp_rank.is_none() {
            degraded_changed |= !state.saw_unranked;
            state.saw_unranked = true;
        }
        let seq = batch.sequence_number;
        let clears = batch.events.iter().any(|event| {
            matches!(&event.data, Some(kv_cache_event::Data::Cleared(cleared)) if cleared.ownership.is_none())
        });
        // Whether some member indexes the batch's rank: its clear reaches
        // the index only then (the restart or cleared gap of a rank no
        // worker stands for says nothing about any member's blocks).
        let taken = state.taken(batch.dp_rank);
        let cursor = state.ranks.entry(rank).or_default();
        // Read before `admit` can drop a pending replay: a restart the gateway
        // notices here loses the replay's batches as a refused resume does.
        let pending = cursor.pending_gap();
        let replaying = cursor.is_replaying();
        let admission = cursor.admit(seq, clears);
        if replaying && !cursor.is_replaying() && !matches!(admission, Admission::Replay { .. }) {
            // This rank's replay from zero reached live (a batch past its
            // cursor, or a clear): the relay's window for it is replayed,
            // so the rounds stop asking for zero on a joiner's account. A
            // gap sighting is not that: the round asks for a replay, and
            // the next one must start from zero again, not from a cursor a
            // multi-rank relay refuses. A joiner's window that still
            // follows arrives on this stream (the relay replays rank by
            // rank); after a break, the next round's cursor is refused by
            // every multi-rank relay (OUT_OF_RANGE) and the engine starts
            // over from zero, as on any reconnect.
            state.history_owed.clear();
        }
        match admission {
            Admission::Apply => {}
            Admission::Stale => {
                debug!(engine = %Endpoint::redacted_str(&state.address), rank, received = seq, "Skipping stale KV event batch");
                state.count_batch(batch.dp_rank, "stale");
                return BatchOutcome::Skipped;
            }
            Admission::Restart if taken => {
                warn!(
                    engine = %Endpoint::redacted_str(&state.address),
                    rank,
                    received = seq,
                    "KV event publisher restarted; clearing its rank's index state"
                );
                // A replay that was pending went with the old publisher, on
                // this rank or on another whose cursor the clear reset: its
                // batches count as missed here, once, as they do when the
                // relay reports the restart as `DATA_LOSS` (`reset_members`).
                Self::clear_rank(
                    indexer,
                    state,
                    batch.dp_rank,
                    pending,
                    ResyncReason::PublisherRestart,
                );
                degraded_changed = true;
            }
            Admission::Restart => {
                // The publisher of a rank no member stands for restarted: its
                // cursor started over in `admit`, and no member's index holds
                // its blocks, so nothing is cleared and the other cursors
                // keep their place. A replay pending on that rank went with
                // its old publisher.
                warn!(
                    engine = %Endpoint::redacted_str(&state.address),
                    rank,
                    received = seq,
                    "KV event publisher of a rank no worker stands for restarted; the index is kept"
                );
                for member in &state.members {
                    Metrics::record_kv_event_missed_batches(&member.url, pending);
                }
                degraded_changed = true;
            }
            Admission::Replay { expected } => {
                // A sighting, not a loss: the replay may bring every batch of
                // the gap, so nothing is counted missed at the request. The
                // loss is counted once where it is known: on the unrecovered
                // outcome, or where a pending replay is dropped
                // (`reset_members` when the resume is refused, `clear_rank`
                // when a publisher starts over or a gap is cleared).
                state.count_gap(batch.dp_rank, "replay_requested", 0);
                return BatchOutcome::Gap {
                    expected,
                    received: seq,
                };
            }
            Admission::Unrecovered { missed, cleared } => {
                warn!(
                    engine = %Endpoint::redacted_str(&state.address),
                    rank,
                    missed,
                    cleared,
                    "KV event gap could not be replayed; continuing from the live stream"
                );
                if cleared && taken {
                    Self::clear_rank(indexer, state, batch.dp_rank, 0, ResyncReason::GapCleared);
                }
                state.count_gap(
                    batch.dp_rank,
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
        if !taken {
            // A rank no worker stands for: its cursor moved and the stream
            // is working, but its blocks are that rank's cache, not any
            // member's. Counted, not indexed.
            state.count_batch(batch.dp_rank, "other_rank");
            if degraded_changed {
                state.publish_degraded();
            }
            return BatchOutcome::Skipped;
        }
        let (mut stored_blocks, mut removed_blocks) = (0usize, 0usize);
        for event in &batch.events {
            match &event.data {
                Some(kv_cache_event::Data::Stored(stored)) => stored_blocks += stored.blocks.len(),
                Some(kv_cache_event::Data::Removed(removed)) => {
                    removed_blocks += removed.block_hashes.len();
                }
                _ => {}
            }
        }
        for member in state.members.iter_mut().filter(|m| m.takes(batch.dp_rank)) {
            let Some(worker_id) = member.worker_id else {
                continue;
            };
            let started = Instant::now();
            let counters_before = member.index.counters;
            Self::apply_events(&batch.events, worker_id, indexer, &mut member.index);
            Metrics::record_kv_event_apply(&member.url, started.elapsed().as_secs_f64());
            Self::record_not_indexed(&member.url, &member.index.counters, &counters_before);
            if stored_blocks > 0 {
                Metrics::record_kv_event_blocks(&member.url, "stored", stored_blocks);
            }
            if removed_blocks > 0 {
                Metrics::record_kv_event_blocks(&member.url, "removed", removed_blocks);
            }
            Self::record_lag(&member.url, batch.timestamp);
            Metrics::record_kv_event_batch(&member.url, "applied");
            Metrics::set_kv_index_blocks(&member.url, indexer.worker_block_count(worker_id));
        }
        if degraded_changed {
            state.publish_degraded();
        }
        BatchOutcome::Applied
    }

    /// A chunk of a relay state snapshot (`KvSnapshotChunk`): the live set the
    /// relay recorded, replacing the engine's state. Chunk 0 clears every
    /// member and every cursor and counts a `snapshot` resync per member;
    /// every chunk is applied outside the admission rules and moves its
    /// rank's cursor to its stamp, which the relay chose so that live events
    /// continue after the last one. The chunk's blocks are accounted like an
    /// applied batch's: counted under `op="snapshot"` (they are the relay's
    /// record, not stores the publisher sent) and reflected in the member's
    /// index-blocks gauge.
    fn admit_snapshot_chunk(
        batch: &KvEventBatch,
        chunk: &KvSnapshotChunk,
        indexer: &KvIndex,
        state: &mut EngineStreamState,
        on_batch: &mut impl FnMut(&KvEventBatch),
    ) -> BatchOutcome {
        let rank = batch.dp_rank.unwrap_or(0);
        if batch.dp_rank.is_none() {
            state.saw_unranked = true;
        }
        if chunk.index == 0 || state.snapshot.is_none() {
            if chunk.index != 0 {
                warn!(
                    engine = %Endpoint::redacted_str(&state.address),
                    rank,
                    index = chunk.index,
                    "KV event relay snapshot arrived without its first chunk; taking it as a resync"
                );
            }
            info!(
                engine = %Endpoint::redacted_str(&state.address),
                rank,
                chunks = chunk.count,
                blocks = chunk.blocks,
                unknown_before = chunk.unknown_before,
                through = batch.sequence_number + u64::from(chunk.count.saturating_sub(chunk.index + 1)),
                "KV event relay served a state snapshot; replacing the engine's index state"
            );
            if chunk.unknown_before > 0 {
                warn!(
                    engine = %Endpoint::redacted_str(&state.address),
                    rank,
                    unknown_before = chunk.unknown_before,
                    "KV event relay snapshot starts late: the engine's blocks from before the \
                     relay's record are unknown; the index is partial until they leave the \
                     engine (rank degraded)"
                );
            }
            for member in &mut state.members {
                if let Some(worker_id) = member.worker_id {
                    Self::apply_cleared(worker_id, indexer, &mut member.index);
                }
                Metrics::record_kv_event_resync(&member.url, ResyncReason::Snapshot.as_str());
            }
            for cursor in state.ranks.values_mut() {
                cursor.reset();
            }
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
        // A chunk of a rank no member stands for takes its share of the
        // snapshot's bookkeeping (the clear above, its rank's cursor, the
        // count below) and none of its blocks.
        let taken = state.taken(batch.dp_rank);
        if taken {
            let mut snapshot_blocks = 0usize;
            for event in &batch.events {
                if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                    snapshot_blocks += stored.blocks.len();
                }
            }
            for member in state.members.iter_mut().filter(|m| m.takes(batch.dp_rank)) {
                let Some(worker_id) = member.worker_id else {
                    continue;
                };
                let counters_before = member.index.counters;
                Self::apply_events(&batch.events, worker_id, indexer, &mut member.index);
                Self::record_not_indexed(&member.url, &member.index.counters, &counters_before);
                if snapshot_blocks > 0 {
                    Metrics::record_kv_event_blocks(&member.url, "snapshot", snapshot_blocks);
                }
                Self::record_lag(&member.url, batch.timestamp);
                Metrics::record_kv_event_batch(&member.url, "snapshot");
                Metrics::set_kv_index_blocks(&member.url, indexer.worker_block_count(worker_id));
            }
        } else {
            state.count_batch(batch.dp_rank, "other_rank");
        }
        if let Some(progress) = &mut state.snapshot {
            progress.applied += 1;
            if progress.applied >= progress.count {
                info!(
                    engine = %Endpoint::redacted_str(&state.address),
                    chunks = progress.count,
                    blocks = progress.blocks,
                    through = batch.sequence_number,
                    "KV event relay snapshot applied; continuing with live events"
                );
                state.snapshot = None;
                // The relay's whole live set is in: every rank's state as of
                // its cut, the joiner's included, so nothing is owed on its
                // account.
                state.history_owed.clear();
                state.publish_degraded();
            }
        }
        if taken {
            BatchOutcome::Applied
        } else {
            BatchOutcome::Skipped
        }
    }

    /// One rank's publisher lost its history (a restart, or an unreplayable
    /// gap too large to keep): the index state of every member that indexes
    /// the rank goes with it, and `pending`, the batches of a replay that
    /// went with the old publisher, count as missed under each, beside the
    /// resync. A whole-engine member pools every rank's blocks in one state,
    /// so its clear also starts the other ranks' cursors over: their next
    /// batch is taken as a first one instead of clearing the index a second
    /// time, and a replay pending on one of them counts as missed too. A
    /// pinned member holds its own rank alone and the other cursors keep
    /// their place: their ranks' restarts clear their own members.
    fn clear_rank(
        indexer: &KvIndex,
        state: &mut EngineStreamState,
        batch_rank: Option<i32>,
        pending: u64,
        reason: ResyncReason,
    ) {
        let rank = batch_rank.unwrap_or(0);
        let whole_engine = state.members.iter().any(|member| member.rank.is_none());
        let mut dropped = 0;
        if whole_engine {
            for (other, cursor) in &mut state.ranks {
                if *other != rank {
                    dropped += cursor.pending_gap();
                    cursor.reset();
                }
            }
        }
        for member in state.members.iter_mut().filter(|m| m.takes(batch_rank)) {
            if let Some(worker_id) = member.worker_id {
                Self::apply_cleared(worker_id, indexer, &mut member.index);
            }
            let missed = pending + if member.rank.is_none() { dropped } else { 0 };
            Metrics::record_kv_event_missed_batches(&member.url, missed);
            Metrics::record_kv_event_resync(&member.url, reason.as_str());
        }
    }

    /// Publish the stores a batch placed without their parent, the sub-block
    /// stores it left out of the index and the stores the index declined at an
    /// arena cap, if any.
    fn record_not_indexed(
        worker_url: &str,
        after: &WorkerIndexCounters,
        before: &WorkerIndexCounters,
    ) {
        let (stores, blocks) = after.parentless_since(before);
        if stores > 0 {
            Metrics::record_kv_event_parentless(worker_url, stores, blocks);
        }
        let (stores, blocks) = after.sub_block_since(before);
        if stores > 0 {
            Metrics::record_kv_event_sub_block(worker_url, stores, blocks);
        }
        let (stores, blocks) = after.cap_refused_since(before);
        if stores > 0 {
            Metrics::record_kv_event_cap_refused(worker_url, stores, blocks);
        }
    }

    /// The server declared its history gone, or the engine starts over for a
    /// member that joined after the stream had delivered: drop every member's
    /// index state and every cursor, so the next stream is taken from
    /// wherever it starts. A replay that was pending is lost with them: its
    /// batches count as missed, once, under each member whose ranks they
    /// were, beside the resync.
    pub(super) fn reset_members(
        indexer: &KvIndex,
        state: &mut EngineStreamState,
        reason: ResyncReason,
    ) {
        let EngineStreamState {
            ranks,
            members,
            saw_unranked,
            ..
        } = state;
        for member in members.iter_mut() {
            let pending: u64 = ranks
                .iter()
                .filter(|(rank, _)| member.counts_rank(**rank, *saw_unranked))
                .map(|(_, cursor)| cursor.pending_gap())
                .sum();
            if let Some(worker_id) = member.worker_id {
                Self::apply_cleared(worker_id, indexer, &mut member.index);
                Metrics::set_kv_index_blocks(&member.url, indexer.worker_block_count(worker_id));
            }
            Metrics::record_kv_event_missed_batches(&member.url, pending);
            Metrics::record_kv_event_resync(&member.url, reason.as_str());
            Metrics::set_kv_event_degraded_ranks(&member.url, 0);
        }
        for cursor in ranks.values_mut() {
            cursor.reset();
        }
        state.snapshot = None;
    }

    /// The engine starts over from zero for the members that joined after
    /// the stream had delivered (`EngineStreamState::joined_late`), and for
    /// them alone: a joiner's rank owes its history (`history_owed`), so
    /// the rounds that follow start from zero until it has a cursor or a
    /// round reached live for an incumbent (`starts_from_zero`), while the
    /// other ranks keep their cursors and index state and take the replay's
    /// batches up to their cursor as stale, so a relay without history
    /// leaves them as a reconnect would. A member that stands for the whole
    /// engine pools every rank, so with one on the stream, or joining,
    /// every member is reset as for a lost history.
    pub(super) fn rejoin_members(indexer: &KvIndex, state: &mut EngineStreamState) {
        let joined = std::mem::take(&mut state.joined_late);
        if joined.iter().any(Option::is_none) || state.members.iter().any(|m| m.rank.is_none()) {
            Self::reset_members(indexer, state, ResyncReason::RankJoined);
            return;
        }
        for rank in joined.iter().flatten() {
            state.ranks.remove(rank);
            if !state.history_owed.contains(rank) {
                state.history_owed.push(*rank);
            }
        }
        for member in state.members.iter().filter(|m| joined.contains(&m.rank)) {
            Metrics::record_kv_event_resync(&member.url, ResyncReason::RankJoined.as_str());
        }
        state.snapshot = None;
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
        kv_event_recovery::{Cursor, GAP_CLEAR_THRESHOLD, RESTART_WINDOW},
        test_support::kv_cache_cleared,
        BasicWorkerBuilder, ConnectionMode,
    };

    /// A gRPC worker object at `url`, for a member of a test engine.
    fn test_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        )
    }

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

    /// One stored block of `width` tokens under `parent`, its tokens unique to `id`.
    fn stored_width(parent: Option<i64>, id: i64, width: i32) -> KvCacheEvent {
        KvCacheEvent {
            event_id: 0,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![KvBlock {
                    block_hash: id,
                    token_ids: (0..width as u32).map(|i| (id as u32) * 64 + i).collect(),
                    block_size: width,
                    ..Default::default()
                }],
                parent_block_hash: parent,
                ..Default::default()
            })),
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
    /// a clear, and costs the lookup path nothing.
    #[test]
    fn index_block_gauge_follows_stores_removals_and_a_clear() {
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
            let mut sim = Sim::new();
            assert_eq!(index_blocks(&handle), None, "nothing applied yet");

            assert_eq!(
                sim.feed(&batch(1, None, vec![stored(None, &[1, 2, 3])])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 3);
            assert_eq!(index_blocks(&handle), Some(3.0), "three blocks stored");

            assert_eq!(
                sim.feed(&batch(2, None, vec![removed(&[3])])),
                BatchOutcome::Applied
            );
            assert_eq!(index_blocks(&handle), Some(2.0), "one removed");

            assert_eq!(
                sim.feed(&batch(3, None, vec![stored(Some(2), &[4, 5])])),
                BatchOutcome::Applied
            );
            assert_eq!(index_blocks(&handle), Some(4.0), "two more stored");

            assert_eq!(
                sim.feed(&batch(4, None, vec![kv_cache_cleared()])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 0);
            assert_eq!(index_blocks(&handle), Some(0.0), "cleared");
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
        state: EngineStreamState,
        reference: ReferenceIndexer,
        /// The reference's block of record: the widest block stored so far.
        block_of_record: i32,
    }

    impl Sim {
        fn new() -> Self {
            let indexer = KvIndex::chain();
            let worker = indexer.intern_worker("grpc://w1:9000").unwrap();
            let member = Member::new(test_worker("grpc://w1:9000"), Some(worker));
            Self {
                indexer,
                worker,
                state: EngineStreamState::new("grpc://w1:9000", vec![member]),
                reference: ReferenceIndexer::new(),
                block_of_record: 0,
            }
        }

        /// The one member stands for `rank` of its engine, as a rank worker
        /// does; the engine's stream names every rank's batches.
        fn pin(&mut self, rank: i32) {
            self.state.members[0].rank = Some(rank);
        }

        /// An engine of `ranks` rank workers, `grpc://w1:9000@<rank>` each,
        /// on one stream; `worker` is rank 0's id.
        fn with_ranks(ranks: i32) -> Self {
            let mut sim = Self::new();
            sim.state.members.clear();
            for rank in 0..ranks {
                let url = format!("grpc://w1:9000@{rank}");
                let worker_id = sim.indexer.intern_worker(&url).unwrap();
                let mut member = Member::new(test_worker(&url), Some(worker_id));
                member.rank = Some(rank);
                sim.state.members.push(member);
            }
            sim.worker = sim.state.members[0].worker_id.unwrap();
            sim
        }

        /// The blocks the index holds for member `at`.
        fn blocks_of(&self, at: usize) -> usize {
            self.indexer
                .worker_block_count(self.state.members[at].worker_id.unwrap())
        }

        /// The one member's degraded ranks.
        fn degraded_ranks(&self) -> usize {
            self.state.degraded_ranks_of(&self.state.members[0])
        }

        /// Feed a batch through the real admission path.
        fn feed(&mut self, b: &KvEventBatch) -> BatchOutcome {
            KvEventMonitor::admit_batch(
                b,
                &self.indexer,
                &mut self.state,
                &mut |_: &KvEventBatch| {},
            )
        }

        /// Apply a batch to the reference with the subscriber's semantics
        /// (a store whose parent is unknown starts a new chain; a store
        /// narrower than the block of record is left out).
        fn reference_apply(&mut self, b: &KvEventBatch) {
            let seed = namespace_seed(None, None);
            for event in &b.events {
                match event.data.as_ref().unwrap() {
                    kv_cache_event::Data::Stored(st) => {
                        let width = st.blocks.first().map_or(0, |block| block.block_size);
                        if width > self.block_of_record {
                            self.block_of_record = width;
                        } else if width > 0 && width < self.block_of_record {
                            continue;
                        }
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
                    kv_cache_event::Data::Cleared(_) => self.reference_clear(),
                }
            }
        }

        /// Feed to both: what a correctly delivered batch does.
        fn deliver(&mut self, b: &KvEventBatch) -> BatchOutcome {
            self.reference_apply(b);
            self.feed(b)
        }

        /// A clear of the reference: its index and its block of record go
        /// together, as production's do on a clear.
        fn reference_clear(&mut self) {
            self.reference.apply_cleared(self.worker);
            self.block_of_record = 0;
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
            sim.deliver(&batch(
                0,
                None,
                vec![kv_cache_cleared(), stored(None, &[1, 2])]
            )),
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

    /// The value of one series of a local recorder: the line that starts
    /// with `name{` and carries every label (the render's label order is
    /// not part of the contract).
    fn series_value(
        handle: &metrics_exporter_prometheus::PrometheusHandle,
        name: &str,
        labels: &[&str],
    ) -> Option<f64> {
        handle
            .render()
            .lines()
            .find(|line| {
                line.starts_with(&format!("{name}{{"))
                    && labels.iter().all(|label| line.contains(label))
            })
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
    }

    /// The gap series of the worker under test, read from a recorder of the
    /// test's own: the gap sightings by outcome, the missed batches and the
    /// resyncs by reason.
    struct GapCounters {
        recorder: metrics_exporter_prometheus::PrometheusRecorder,
        handle: metrics_exporter_prometheus::PrometheusHandle,
    }

    impl GapCounters {
        const WORKER: &'static str = "worker=\"grpc://w1:9000\"";

        fn new() -> Self {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            Self { recorder, handle }
        }

        /// `smg_kv_event_gaps_total` for `outcome`, if the series exists.
        fn gaps(&self, outcome: &str) -> Option<f64> {
            series_value(
                &self.handle,
                "smg_kv_event_gaps_total",
                &[Self::WORKER, &format!("outcome=\"{outcome}\"")],
            )
        }

        /// `smg_kv_event_missed_batches_total`, if the series exists.
        fn missed(&self) -> Option<f64> {
            series_value(
                &self.handle,
                "smg_kv_event_missed_batches_total",
                &[Self::WORKER],
            )
        }

        /// `smg_kv_event_resyncs_total` for `reason`, if the series exists.
        fn resyncs(&self, reason: &str) -> Option<f64> {
            series_value(
                &self.handle,
                "smg_kv_event_resyncs_total",
                &[Self::WORKER, &format!("reason=\"{reason}\"")],
            )
        }
    }

    /// A gap the publisher replays in full loses no batch: the sighting is
    /// counted under `outcome="replay_requested"` and
    /// `smg_kv_event_missed_batches_total` stays at zero.
    #[test]
    fn a_replayed_gap_counts_no_missed_batches() {
        let counters = GapCounters::new();
        metrics::with_local_recorder(&counters.recorder, || {
            // The subscription start creates the worker's series at zero.
            Metrics::init_kv_event_series("grpc://w1:9000");
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
            let at_request = counters.missed();
            // Resuming after 2, the server replays 3 and continues live.
            assert_eq!(sim.deliver(&b3), BatchOutcome::Applied);
            assert_eq!(sim.deliver(&b4), BatchOutcome::Applied);
            assert_eq!(
                sim.deliver(&batch(5, None, vec![stored(Some(7), &[8])])),
                BatchOutcome::Applied
            );
            assert_eq!(counters.gaps("replay_requested"), Some(1.0), "the sighting");
            assert_eq!(counters.gaps("unrecovered_kept"), Some(0.0));
            assert_eq!(counters.gaps("unrecovered_cleared"), Some(0.0));
            assert_eq!(
                [at_request, counters.missed()],
                [Some(0.0), Some(0.0)],
                "missed batches at the request and after the replay"
            );
            sim.assert_matches_reference();
        });
    }

    /// A gap the publisher cannot replay counts its lost batches once, on
    /// the unrecovered outcome: neither the request before it nor a later
    /// gap's request adds to `smg_kv_event_missed_batches_total`.
    #[test]
    fn an_unrecovered_gap_counts_its_missed_batches_once() {
        let counters = GapCounters::new();
        metrics::with_local_recorder(&counters.recorder, || {
            Metrics::init_kv_event_series("grpc://w1:9000");
            let mut sim = Sim::new();
            sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
            sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
            // Batches 3 and 4 are lost; the server has no history: after the
            // replay request it streams 5 again.
            let b5 = batch(5, None, vec![stored(Some(3), &[9])]);
            assert_eq!(
                sim.feed(&b5),
                BatchOutcome::Gap {
                    expected: 3,
                    received: 5
                }
            );
            let at_request = counters.missed();
            assert_eq!(sim.deliver(&b5), BatchOutcome::Applied);
            let after_the_skip = counters.missed();
            assert_eq!(counters.gaps("replay_requested"), Some(1.0));
            assert_eq!(counters.gaps("unrecovered_kept"), Some(1.0));
            assert_eq!(counters.gaps("unrecovered_cleared"), Some(0.0));
            // A later gap is a sighting again: a fresh replay request, no loss yet.
            assert_eq!(
                sim.feed(&batch(7, None, vec![])),
                BatchOutcome::Gap {
                    expected: 6,
                    received: 7
                }
            );
            assert_eq!(counters.gaps("replay_requested"), Some(2.0));
            assert_eq!(
                [at_request, after_the_skip, counters.missed()],
                [Some(0.0), Some(2.0), Some(2.0)],
                "missed batches at the request, once the stream skipped ahead, after a later request"
            );
        });
    }

    /// A replay the server refuses (`OUT_OF_RANGE` on the resume, or
    /// `DATA_LOSS` on the stream) never reaches an unrecovered outcome: the
    /// subscriber drops the worker's state, and the batches the pending
    /// replay was asked for count as missed there, once, beside the resync.
    #[test]
    fn a_refused_replay_counts_its_pending_gap_as_missed_once() {
        let counters = GapCounters::new();
        metrics::with_local_recorder(&counters.recorder, || {
            Metrics::init_kv_event_series("grpc://w1:9000");
            let mut sim = Sim::new();
            sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
            sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
            // Batches 3 to 5 are lost; 6 arrives and a replay from 3 is asked for.
            assert_eq!(
                sim.feed(&batch(6, None, vec![stored(Some(3), &[9])])),
                BatchOutcome::Gap {
                    expected: 3,
                    received: 6
                }
            );
            assert_eq!(
                counters.missed(),
                Some(0.0),
                "nothing is known lost at the request"
            );
            // The server no longer holds 3 and refuses the resume: the
            // subscriber drops the worker's state, the pending replay with it.
            KvEventMonitor::reset_members(&sim.indexer, &mut sim.state, ResyncReason::OutOfRange);
            assert_eq!(sim.state.ranks[&0].replay_pending(), None);
            assert_eq!(counters.resyncs("out_of_range"), Some(1.0));
            assert_eq!(
                counters.missed(),
                Some(3.0),
                "the three batches the refused replay would have brought, once"
            );
        });
    }

    /// A publisher restart the gateway notices itself (a sequence below the
    /// cursor on a fresh connection) drops a pending replay as a refused
    /// resume does: the batches it was asked for count as missed there, once,
    /// beside the restart.
    #[test]
    fn a_restart_the_gateway_notices_counts_the_pending_gap_as_missed_once() {
        let counters = GapCounters::new();
        metrics::with_local_recorder(&counters.recorder, || {
            Metrics::init_kv_event_series("grpc://w1:9000");
            let mut sim = Sim::new();
            sim.deliver(&batch(1, None, vec![stored(None, &[1, 2])]));
            sim.deliver(&batch(2, None, vec![stored(Some(2), &[3])]));
            // Batches 3 to 5 are lost; 6 arrives and a replay from 3 is asked for.
            assert_eq!(
                sim.feed(&batch(6, None, vec![stored(Some(3), &[9])])),
                BatchOutcome::Gap {
                    expected: 3,
                    received: 6
                }
            );
            assert_eq!(
                counters.missed(),
                Some(0.0),
                "nothing is known lost at the request"
            );
            // The worker came back before any replay: the stream reconnected
            // and the new publisher counts from 1 with an empty cache.
            sim.state.ranks.get_mut(&0).unwrap().reconnected();
            let fresh = batch(1, None, vec![stored(None, &[7])]);
            sim.reference_clear();
            sim.reference_apply(&fresh);
            assert_eq!(sim.feed(&fresh), BatchOutcome::Applied);
            assert_eq!(sim.state.ranks[&0].replay_pending(), None);
            assert_eq!(counters.resyncs("publisher_restart"), Some(1.0));
            assert_eq!(
                counters.missed(),
                Some(3.0),
                "the three batches the dropped replay would have brought, once"
            );
            sim.assert_matches_reference();
        });
    }

    /// A restart one rank's stream shows (a sequence below its cursor on a
    /// fresh connection) starts the other ranks' cursors over with the index;
    /// a replay pending on one of them goes with its cursor, and the batches
    /// it was asked for count as missed there, once, as the restarting rank's
    /// own would.
    #[test]
    fn a_restart_another_rank_notices_counts_a_pending_gap_as_missed_once() {
        let counters = GapCounters::new();
        metrics::with_local_recorder(&counters.recorder, || {
            Metrics::init_kv_event_series("grpc://w1:9000");
            let mut sim = Sim::new();
            // Two publishers, each numbering from 1, interleaved on one stream.
            sim.deliver(&batch(1, Some(0), vec![stored(None, &[1, 2])]));
            sim.deliver(&batch(2, Some(0), vec![stored(Some(2), &[3])]));
            sim.deliver(&batch(1, Some(1), vec![stored(None, &[101, 102])]));
            sim.deliver(&batch(2, Some(1), vec![stored(Some(102), &[103])]));
            // Rank 1's batches 3 to 5 are lost; 6 arrives and a replay from 3
            // is asked for.
            assert_eq!(
                sim.feed(&batch(6, Some(1), vec![stored(Some(103), &[109])])),
                BatchOutcome::Gap {
                    expected: 3,
                    received: 6
                }
            );
            assert_eq!(
                counters.missed(),
                Some(0.0),
                "nothing is known lost at the request"
            );
            // The engine came back before any replay: the stream reconnected
            // and rank 0's new publisher counts from 1 with an empty cache.
            sim.state.reconnected();
            let fresh = batch(1, Some(0), vec![stored(None, &[7])]);
            sim.reference_clear();
            sim.reference_apply(&fresh);
            assert_eq!(sim.feed(&fresh), BatchOutcome::Applied);
            assert_eq!(
                sim.state.ranks[&1].replay_pending(),
                None,
                "rank 1's cursor started over with the index"
            );
            assert_eq!(counters.resyncs("publisher_restart"), Some(1.0));
            assert_eq!(
                counters.missed(),
                Some(3.0),
                "the three batches rank 1's dropped replay would have brought, once"
            );
            sim.assert_matches_reference();
        });
    }

    /// A gap one rank cannot replay and too large to keep (`Unrecovered {
    /// cleared: true }`) clears the worker and starts the other ranks'
    /// cursors over with the index; a replay pending on one of them goes with
    /// its cursor, and the batches it was asked for count as missed beside
    /// the clearing rank's own gap, once.
    #[test]
    fn a_cleared_gap_on_another_rank_counts_a_pending_gap_as_missed_once() {
        let counters = GapCounters::new();
        metrics::with_local_recorder(&counters.recorder, || {
            Metrics::init_kv_event_series("grpc://w1:9000");
            let mut sim = Sim::new();
            sim.deliver(&batch(1, Some(0), vec![stored(None, &[1, 2])]));
            sim.deliver(&batch(1, Some(1), vec![stored(None, &[101, 102])]));
            // Rank 1's batches 2 to 4 are lost; 5 arrives and a replay from 2
            // is asked for.
            assert_eq!(
                sim.feed(&batch(5, Some(1), vec![stored(Some(102), &[105])])),
                BatchOutcome::Gap {
                    expected: 2,
                    received: 5
                }
            );
            // Rank 0's batches 2 and 3 are lost too; 4 arrives and a replay
            // from 2 is asked for.
            assert_eq!(
                sim.feed(&batch(4, Some(0), vec![stored(Some(2), &[4])])),
                BatchOutcome::Gap {
                    expected: 2,
                    received: 4
                }
            );
            assert_eq!(
                counters.missed(),
                Some(0.0),
                "nothing is known lost at the requests"
            );
            // The server skips far ahead on rank 0 instead of replaying: a gap
            // too large to keep, the worker is cleared, every rank's cursor
            // starts over with the index.
            let far = 2 + GAP_CLEAR_THRESHOLD + 1;
            let skip = batch(far, Some(0), vec![stored(None, &[7])]);
            sim.reference_clear();
            sim.reference_apply(&skip);
            assert_eq!(sim.feed(&skip), BatchOutcome::Applied);
            assert_eq!(
                sim.state.ranks[&1].replay_pending(),
                None,
                "rank 1's cursor started over with the index"
            );
            assert_eq!(counters.resyncs("gap_cleared"), Some(1.0));
            assert_eq!(
                counters.missed(),
                Some((far - 2) as f64 + 3.0),
                "rank 0's own gap and the three batches rank 1's dropped replay would have brought, once"
            );
            sim.assert_matches_reference();
        });
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
        sim.reference_clear();
        sim.reference_apply(&fresh);
        assert_eq!(sim.feed(&fresh), BatchOutcome::Applied);
        assert_eq!(
            sim.deliver(&batch(1, None, vec![stored(Some(22), &[23])])),
            BatchOutcome::Applied
        );
        assert!(!sim.state.ranks[&0].is_degraded());
        sim.assert_matches_reference();
    }

    /// The engine restarts with a narrower block than it stored before: the
    /// clear takes the block of record with the blocks, so the fresh stores
    /// are indexed as they come (none counted as a sub-block store) and
    /// production matches the reference, whose record resets the same way.
    #[test]
    fn recovery_publisher_restart_with_a_narrower_block_matches_reference() {
        let mut sim = Sim::new();
        sim.deliver(&batch(
            1,
            None,
            vec![stored_width(None, 1, 32), stored_width(Some(1), 2, 32)],
        ));
        for seq in 2..=(RESTART_WINDOW + 5) {
            sim.deliver(&batch(seq, None, vec![]));
        }
        assert_eq!(sim.state.members[0].index.counters.sub_block_stores, 0);
        // The engine restarts at a 4-token block: its cache is empty and it
        // counts from 0 again.
        let fresh = batch(
            0,
            None,
            vec![stored_width(None, 11, 4), stored_width(Some(11), 12, 4)],
        );
        sim.reference_clear();
        sim.reference_apply(&fresh);
        assert_eq!(sim.feed(&fresh), BatchOutcome::Applied);
        assert_eq!(
            sim.indexer.worker_block_count(sim.worker),
            2,
            "the narrower blocks are indexed"
        );
        assert_eq!(sim.state.members[0].index.counters.sub_block_stores, 0);
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
        sim.reference_clear();
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
        let first = batch(0, None, vec![kv_cache_cleared(), stored(None, &[9])]);
        sim.reference_clear();
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
        sim.deliver(&batch(2, None, vec![kv_cache_cleared()]));
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
        assert_eq!(sim.degraded_ranks(), 0);
        assert_eq!(sim.state.resume_sequence(), 3);
        sim.assert_matches_reference();
        // Rank 1's publisher restarts: the worker's pooled index state is
        // cleared and rank 0's cursor starts over, so its next batch is taken
        // as a first one rather than clearing the index again.
        sim.state.reconnected();
        let fresh = batch(1, Some(1), vec![stored(None, &[111])]);
        sim.reference_clear();
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
        sim.reference_clear();
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
                    events.push(kv_cache_cleared());
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

    /// Under `--dp-aware` an engine's stream names every rank's batches. A
    /// rank worker runs them all through admission (each rank's cursor, so
    /// the resubscribe cursor and the gap detection are the unpinned
    /// worker's) and indexes its own rank's and the unranked ones only.
    #[test]
    fn another_ranks_batch_moves_its_cursor_and_stays_out_of_the_index() {
        let mut sim = Sim::new();
        sim.pin(1);
        assert_eq!(
            sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])])),
            BatchOutcome::Skipped
        );
        assert_eq!(sim.indexer.worker_block_count(sim.worker), 0);
        assert_eq!(sim.state.ranks[&0].cursor(), Cursor::Live(1));
        assert_eq!(
            sim.state.resume_sequence(),
            1,
            "the resubscribe cursor is rank 0's, moved by rank 0's batch"
        );
        assert_eq!(
            sim.feed(&batch(1, Some(1), vec![stored(None, &[3, 4])])),
            BatchOutcome::Applied
        );
        assert_eq!(
            sim.feed(&batch(2, None, vec![stored(None, &[5])])),
            BatchOutcome::Applied
        );
        assert_eq!(sim.indexer.worker_block_count(sim.worker), 3);
        // Rank 0's next batch continues on its cursor: no gap, not indexed.
        assert_eq!(
            sim.feed(&batch(3, Some(0), vec![stored(None, &[6])])),
            BatchOutcome::Skipped
        );
        assert_eq!(sim.state.resume_sequence(), 3);
        assert_eq!(sim.indexer.worker_block_count(sim.worker), 3);
    }

    /// A relay snapshot is one sequence over every rank: a rank worker runs
    /// every chunk through the snapshot's bookkeeping (chunk 0's clear, each
    /// chunk's cursor, the count) and indexes its own rank's chunks only.
    #[test]
    fn another_ranks_snapshot_chunk_counts_toward_the_snapshot_and_stays_out_of_the_index() {
        let mut sim = Sim::new();
        sim.pin(1);
        let mut chunks = snapshot_chunks(
            5,
            Some(0),
            &[vec![stored(None, &[1, 2, 3])], vec![stored(None, &[4, 5])]],
            5,
        );
        chunks[1].dp_rank = Some(1);
        assert_eq!(sim.feed(&chunks[0]), BatchOutcome::Skipped);
        assert!(
            sim.state.snapshot.is_some(),
            "the first chunk opened the snapshot"
        );
        assert_eq!(sim.indexer.worker_block_count(sim.worker), 0);
        assert_eq!(sim.feed(&chunks[1]), BatchOutcome::Applied);
        assert!(sim.state.snapshot.is_none(), "both chunks counted");
        assert_eq!(sim.indexer.worker_block_count(sim.worker), 2);
        assert_eq!(sim.state.ranks[&0].cursor(), Cursor::Live(4));
        assert_eq!(sim.state.ranks[&1].cursor(), Cursor::Live(5));
        assert_eq!(sim.state.resume_sequence(), 4);
    }

    /// Another rank's publisher restart says nothing about this worker's
    /// blocks: under a rank-1 state, rank 0's restart leaves the index and
    /// rank 1's cursor in place and records no resync, while rank 1's own
    /// restart clears the worker once, as on an unpinned worker.
    #[test]
    fn another_ranks_restart_leaves_this_ranks_index_and_cursor() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let worker = "worker=\"grpc://w1:9000\"";
        let restarts = || {
            series_value(
                &handle,
                "smg_kv_event_resyncs_total",
                &[worker, "reason=\"publisher_restart\""],
            )
        };
        metrics::with_local_recorder(&recorder, || {
            Metrics::init_kv_event_series("grpc://w1:9000");
            let mut sim = Sim::new();
            sim.pin(1);
            sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])]));
            sim.feed(&batch(2, Some(0), vec![stored(Some(2), &[3])]));
            assert_eq!(
                sim.feed(&batch(1, Some(1), vec![stored(None, &[101, 102])])),
                BatchOutcome::Applied
            );
            assert_eq!(
                sim.feed(&batch(2, Some(1), vec![stored(Some(102), &[103])])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 3);
            // Rank 0's publisher restarts: a sequence below its cursor on a
            // fresh connection.
            sim.state.reconnected();
            assert_eq!(
                sim.feed(&batch(1, Some(0), vec![stored(None, &[7])])),
                BatchOutcome::Skipped
            );
            assert_eq!(
                sim.indexer.worker_block_count(sim.worker),
                3,
                "rank 1's blocks stay in the index"
            );
            assert_eq!(
                sim.state.ranks[&1].cursor(),
                Cursor::Live(2),
                "rank 1's cursor keeps its place"
            );
            assert_eq!(
                sim.state.resume_sequence(),
                1,
                "rank 0's cursor started over"
            );
            assert_eq!(restarts(), Some(0.0), "no resync of this worker");
            assert_eq!(
                sim.degraded_ranks(),
                0,
                "rank 0's cursor is not this worker's gauge"
            );
            // Rank 1 continues on its cursor: no gap, indexed.
            assert_eq!(
                sim.feed(&batch(3, Some(1), vec![stored(Some(103), &[104])])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 4);
            // Rank 1's own restart clears the worker, once.
            sim.state.reconnected();
            assert_eq!(
                sim.feed(&batch(1, Some(1), vec![stored(None, &[111])])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 1);
            assert_eq!(restarts(), Some(1.0));
        });
    }

    /// A batch that names no rank is every rank's: a pinned worker indexes it
    /// on rank 0's cursor, so that cursor joins the worker's degraded-ranks
    /// gauge once such a batch has ridden it, and not before.
    #[test]
    fn an_unranked_batch_puts_rank_zeros_cursor_in_a_pinned_workers_gauge() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let gauge = || {
            series_value(
                &handle,
                "smg_kv_event_degraded_ranks",
                &["worker=\"grpc://w1:9000\""],
            )
        };
        metrics::with_local_recorder(&recorder, || {
            Metrics::init_kv_event_series("grpc://w1:9000");
            let mut sim = Sim::new();
            sim.pin(1);
            assert_eq!(
                sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])])),
                BatchOutcome::Skipped
            );
            sim.state.ranks.get_mut(&0).unwrap().mark_degraded();
            assert_eq!(
                sim.degraded_ranks(),
                0,
                "rank 0's own batches are not this worker's"
            );
            assert_eq!(gauge(), Some(0.0));
            assert_eq!(
                sim.feed(&batch(2, None, vec![stored(None, &[3])])),
                BatchOutcome::Applied
            );
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 1);
            assert_eq!(
                sim.degraded_ranks(),
                1,
                "an unranked batch rode rank 0's degraded cursor into this worker's index"
            );
            assert_eq!(
                gauge(),
                Some(1.0),
                "the gauge is republished on the batch that put the cursor in"
            );
        });
    }

    /// On an engine's one stream a batch enters the index state of the rank
    /// worker whose rank it names, an unranked batch every member's, and a
    /// batch of a rank no worker stands for none; every batch moves its own
    /// rank's cursor once.
    #[test]
    fn a_batch_enters_its_ranks_member_alone() {
        let mut sim = Sim::with_ranks(2);
        assert_eq!(
            sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])])),
            BatchOutcome::Applied
        );
        assert_eq!((sim.blocks_of(0), sim.blocks_of(1)), (2, 0));
        assert_eq!(
            sim.feed(&batch(1, Some(1), vec![stored(None, &[3])])),
            BatchOutcome::Applied
        );
        assert_eq!((sim.blocks_of(0), sim.blocks_of(1)), (2, 1));
        assert_eq!(
            sim.feed(&batch(2, None, vec![stored(None, &[4])])),
            BatchOutcome::Applied,
            "an unranked batch is every rank's"
        );
        assert_eq!((sim.blocks_of(0), sim.blocks_of(1)), (3, 2));
        assert_eq!(
            sim.feed(&batch(1, Some(5), vec![stored(None, &[5])])),
            BatchOutcome::Skipped,
            "a rank no worker stands for enters no index"
        );
        assert_eq!((sim.blocks_of(0), sim.blocks_of(1)), (3, 2));
        assert_eq!(sim.state.ranks[&0].cursor(), Cursor::Live(2));
        assert_eq!(sim.state.ranks[&1].cursor(), Cursor::Live(1));
        assert_eq!(sim.state.ranks[&5].cursor(), Cursor::Live(1));
        assert_eq!(sim.state.resume_sequence(), 2);
    }

    /// A rank's publisher restart clears the member that stands for that
    /// rank and no other: the other members keep their blocks and their
    /// ranks' cursors keep their place, so their own restarts are still
    /// seen as such.
    #[test]
    fn a_ranks_restart_clears_its_own_member_only() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let restarts = |rank: i32| {
            series_value(
                &handle,
                "smg_kv_event_resyncs_total",
                &[
                    &format!("worker=\"grpc://w1:9000@{rank}\""),
                    "reason=\"publisher_restart\"",
                ],
            )
        };
        metrics::with_local_recorder(&recorder, || {
            Metrics::init_kv_event_series("grpc://w1:9000@0");
            Metrics::init_kv_event_series("grpc://w1:9000@1");
            let mut sim = Sim::with_ranks(2);
            sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])]));
            sim.feed(&batch(2, Some(0), vec![stored(Some(2), &[3])]));
            sim.feed(&batch(1, Some(1), vec![stored(None, &[101, 102])]));
            sim.feed(&batch(2, Some(1), vec![stored(Some(102), &[103])]));
            assert_eq!((sim.blocks_of(0), sim.blocks_of(1)), (3, 3));
            // Rank 1's publisher restarts: a sequence below its cursor on a
            // fresh connection.
            sim.state.reconnected();
            assert_eq!(
                sim.feed(&batch(1, Some(1), vec![stored(None, &[111])])),
                BatchOutcome::Applied
            );
            assert_eq!(
                (sim.blocks_of(0), sim.blocks_of(1)),
                (3, 1),
                "rank 1 starts over, rank 0 keeps its blocks"
            );
            assert_eq!(sim.state.ranks[&0].cursor(), Cursor::Live(2));
            assert_eq!((restarts(0), restarts(1)), (Some(0.0), Some(1.0)));
            // Rank 0 continues on its cursor: no gap, indexed.
            assert_eq!(
                sim.feed(&batch(3, Some(0), vec![stored(Some(3), &[4])])),
                BatchOutcome::Applied
            );
            assert_eq!((sim.blocks_of(0), sim.blocks_of(1)), (4, 1));
            // Rank 0's own restart is seen as such, with no second fresh
            // connection.
            assert_eq!(
                sim.feed(&batch(
                    1,
                    Some(0),
                    vec![kv_cache_cleared(), stored(None, &[7])]
                )),
                BatchOutcome::Applied
            );
            assert_eq!((sim.blocks_of(0), sim.blocks_of(1)), (1, 1));
            assert_eq!((restarts(0), restarts(1)), (Some(1.0), Some(1.0)));
        });
    }

    /// A rank that joined late and publishes nothing owes its history until
    /// a round from zero reached live for an incumbent (a replay that walks
    /// past its cursor; one that only reaches it may still be replaying) or
    /// a relay snapshot came in whole; the joiner's window, if any, follows
    /// on the stream or through the refused cursor's restart. The next
    /// round then resumes from the cursor, and a sequence below it is a
    /// publisher restart again.
    #[test]
    fn an_idle_late_rank_owes_its_history_until_a_round_walked_past_the_join() {
        let mut sim = Sim::with_ranks(2);
        sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])]));
        sim.feed(&batch(2, Some(0), vec![stored(Some(2), &[3])]));
        // Rank 1 joined late: the rounds start from zero while it is owed.
        sim.state.history_owed.push(1);
        assert!(sim.state.starts_from_zero());
        // A replay that ends at rank 0's cursor leaves the history owed: the
        // join may still lie ahead of it.
        assert_eq!(
            sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])])),
            BatchOutcome::Skipped
        );
        assert_eq!(
            sim.feed(&batch(2, Some(0), vec![stored(Some(2), &[3])])),
            BatchOutcome::Skipped
        );
        assert!(
            sim.state.starts_from_zero(),
            "a replay up to the cursor has not walked past the join"
        );
        // The next replay walks past it: the history before the join is
        // delivered, and rank 1, still silent, had none.
        for seq in 1..=2 {
            assert_eq!(
                sim.feed(&batch(seq, Some(0), vec![])),
                BatchOutcome::Skipped
            );
        }
        assert_eq!(
            sim.feed(&batch(3, Some(0), vec![stored(Some(3), &[4])])),
            BatchOutcome::Applied
        );
        assert!(sim.state.history_owed.is_empty());
        assert!(
            !sim.state.starts_from_zero(),
            "the next round resumes from the cursor"
        );
        assert_eq!(sim.state.resume_sequence(), 3);
        // Restart detection is back: a sequence below the cursor on the
        // resumed connection clears rank 0.
        sim.state.reconnected();
        assert_eq!(
            sim.feed(&batch(1, Some(0), vec![stored(None, &[7])])),
            BatchOutcome::Applied
        );
        assert_eq!((sim.blocks_of(0), sim.blocks_of(1)), (1, 0));

        // A relay whose history rolled answers the round from zero with a
        // snapshot: applied whole, it is the history before the join.
        let mut sim = Sim::with_ranks(2);
        sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])]));
        sim.state.history_owed.push(1);
        assert!(sim.state.starts_from_zero());
        let chunks = snapshot_chunks(
            5,
            Some(0),
            &[vec![stored(None, &[1, 2, 3])], vec![stored(Some(3), &[4])]],
            4,
        );
        assert_eq!(sim.feed(&chunks[0]), BatchOutcome::Applied);
        assert_eq!(
            sim.state.history_owed,
            vec![1],
            "a snapshot still arriving is a partial live set"
        );
        assert_eq!(sim.feed(&chunks[1]), BatchOutcome::Applied);
        assert!(sim.state.history_owed.is_empty());
        assert!(!sim.state.starts_from_zero());
        assert_eq!(sim.state.resume_sequence(), 5);
    }

    /// A gap in the round from zero (the first batch past an incumbent's
    /// cursor skips ahead) asks for a replay and keeps the history owed:
    /// the next round starts from zero again, where the repeated gap
    /// settles as an unrecovered one with the incumbent's blocks kept,
    /// instead of a resume from the cursor, which a multi-rank relay
    /// refuses and answers with a reset of every member.
    #[test]
    fn a_gap_in_the_round_from_zero_keeps_the_history_owed() {
        let mut sim = Sim::with_ranks(2);
        sim.feed(&batch(1, Some(0), vec![stored(None, &[1, 2])]));
        sim.feed(&batch(2, Some(0), vec![stored(Some(2), &[3])]));
        sim.state.history_owed.push(1);
        assert!(sim.state.starts_from_zero());
        for seq in 1..=2 {
            assert_eq!(
                sim.feed(&batch(seq, Some(0), vec![])),
                BatchOutcome::Skipped
            );
        }
        // The stream skips 3: a gap sighting past the cursor, not the
        // replay reaching live.
        assert_eq!(
            sim.feed(&batch(4, Some(0), vec![stored(Some(3), &[5])])),
            BatchOutcome::Gap {
                expected: 3,
                received: 4
            }
        );
        assert_eq!(
            sim.state.history_owed,
            vec![1],
            "a gap sighting keeps the history owed"
        );
        assert!(
            sim.state.starts_from_zero(),
            "the next round starts from zero again"
        );
        for seq in 1..=2 {
            assert_eq!(
                sim.feed(&batch(seq, Some(0), vec![])),
                BatchOutcome::Skipped
            );
        }
        // The relay skips 3 again: the gap settles as unrecovered, the
        // blocks are kept and the history is delivered.
        assert_eq!(
            sim.feed(&batch(4, Some(0), vec![stored(Some(3), &[5])])),
            BatchOutcome::Applied
        );
        assert!(sim.state.ranks[&0].is_degraded());
        assert_eq!(sim.blocks_of(0), 4, "the incumbent keeps its blocks");
        assert!(sim.state.history_owed.is_empty());
        assert!(!sim.state.starts_from_zero());
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
        assert_eq!(sim.degraded_ranks(), 1);
        assert_eq!(
            sim.feed(&batch(10, None, vec![stored(Some(1), &[2])])),
            BatchOutcome::Applied
        );
        assert!(sim.state.ranks[&0].is_degraded(), "until the next resync");
        let whole = snapshot_chunks(12, None, &[vec![stored(None, &[1])]], 1);
        assert_eq!(sim.feed(&whole[0]), BatchOutcome::Applied);
        assert!(!sim.state.ranks[&0].is_degraded());
        assert_eq!(sim.degraded_ranks(), 0);
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
            assert_eq!(sim.state.members[0].index.counters.parentless_stores, 1);
            // A chained store after a held parent is not counted.
            sim.deliver(&batch(3, None, vec![stored(Some(2), &[6])]));
            assert_eq!(
                counter(&handle, "smg_kv_event_parentless_stores_total"),
                Some(1.0)
            );
            sim.assert_matches_reference();
        });
    }

    /// A store the index declines at an arena cap is counted per worker from
    /// the block declined, on either attempt. A run's child table at its cap
    /// (2^18 slots hold 196,608 live children) is filled by two-block chains
    /// sharing their first block, the churn bench's fan-out shape; the next
    /// chain's first block is placed and its second declined (one store, one
    /// block), and a chain under a parent never seen takes the same refusal on
    /// the parent-less retry. The reference has no cap, so this feeds the real
    /// path alone.
    #[test]
    fn cap_refusals_are_counted_per_worker_on_either_attempt() {
        use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

        fn counter(handle: &PrometheusHandle, name: &str) -> Option<f64> {
            handle
                .render()
                .lines()
                .find(|line| line.starts_with(&format!("{name}{{worker=\"grpc://w1:9000\"}}")))
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|value| value.parse().ok())
        }

        // The first chain's second block is the run's own; every later chain is a
        // child of the run.
        const CHAINS: i64 = 196_609;
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let mut sim = Sim::new();
            let fill: Vec<KvCacheEvent> = (0..CHAINS)
                .map(|chain| stored(None, &[1, 1_000 + chain]))
                .collect();
            sim.feed(&batch(1, None, fill));
            assert_eq!(
                counter(&handle, "smg_kv_event_cap_refused_stores_total"),
                None,
                "the table takes the children up to the cap"
            );
            assert_eq!(
                sim.indexer.worker_block_count(sim.worker),
                CHAINS as usize + 1
            );
            // The first block is placed, the second declined: one block refused.
            sim.feed(&batch(2, None, vec![stored(None, &[1, 2_000_000])]));
            assert_eq!(
                counter(&handle, "smg_kv_event_cap_refused_stores_total"),
                Some(1.0)
            );
            assert_eq!(
                counter(&handle, "smg_kv_event_cap_refused_blocks_total"),
                Some(1.0)
            );
            assert_eq!(sim.state.members[0].index.counters.cap_refused_stores, 1);
            assert_eq!(sim.state.members[0].index.counters.cap_refused_blocks, 1);
            // Under a parent never seen, the parent-less retry is declined the same
            // way and counted on both counters.
            sim.feed(&batch(3, None, vec![stored(Some(99), &[1, 2_000_001])]));
            assert_eq!(
                counter(&handle, "smg_kv_event_cap_refused_stores_total"),
                Some(2.0)
            );
            assert_eq!(
                counter(&handle, "smg_kv_event_cap_refused_blocks_total"),
                Some(2.0)
            );
            assert_eq!(
                counter(&handle, "smg_kv_event_parentless_stores_total"),
                Some(1.0)
            );
            // The declined blocks are held by nothing; the placed first block was
            // held already.
            assert_eq!(
                sim.indexer.worker_block_count(sim.worker),
                CHAINS as usize + 1
            );
        });
    }

    /// A store narrower than the block of record (a hybrid engine's prefix
    /// entry at its hash granularity, under a parent it never publishes) is
    /// counted under its own metric and never as parent-less, and the index
    /// holds what the full blocks made it.
    #[test]
    fn sub_block_stores_are_counted_and_not_indexed() {
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
            // Two full blocks, then the tail entry under a parent never
            // published, in the one batch the engine sends them in.
            sim.deliver(&batch(
                1,
                None,
                vec![
                    stored_width(None, 1, 32),
                    stored_width(Some(1), 2, 32),
                    stored_width(Some(999), 3, 4),
                ],
            ));
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 2);
            assert_eq!(
                counter(&handle, "smg_kv_event_sub_block_stores_total"),
                Some(1.0)
            );
            assert_eq!(
                counter(&handle, "smg_kv_event_sub_block_blocks_total"),
                Some(1.0)
            );
            assert_eq!(
                counter(&handle, "smg_kv_event_parentless_stores_total"),
                None
            );
            assert_eq!(sim.state.members[0].index.counters.sub_block_stores, 1);
            assert_eq!(sim.state.members[0].index.counters.parentless_stores, 0);
            // The chain continues from the full block.
            sim.deliver(&batch(2, None, vec![stored_width(Some(2), 4, 32)]));
            assert_eq!(sim.indexer.worker_block_count(sim.worker), 3);
            assert_eq!(
                counter(&handle, "smg_kv_event_parentless_stores_total"),
                None
            );
            sim.assert_matches_reference();
        });
    }

    /// A hybrid engine whose hash granularity is narrower than its block
    /// registers a prompt's tail entry at that granularity once the request's
    /// computed tokens reach the next block boundary: in the step that stores
    /// the full block, a one-hash store of a fraction of the block under an
    /// intermediate hash the engine never publishes. Its eviction later lists
    /// the entry beside the full hash. The index keeps the full blocks only:
    /// no parent-less store, no root chain of one entry.
    #[test]
    fn a_tail_entry_beside_a_full_block_under_a_wider_record_is_not_parent_less() {
        fn run(parent: Option<i64>, ids: &[i64], width: u32) -> KvCacheEvent {
            KvCacheEvent {
                event_id: 0,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: ids
                        .iter()
                        .map(|&id| KvBlock {
                            block_hash: id,
                            token_ids: (0..width).map(|i| (id as u32) * 1000 + i).collect(),
                            block_size: width as i32,
                            ..Default::default()
                        })
                        .collect(),
                    parent_block_hash: parent,
                    ..Default::default()
                })),
            }
        }
        let mut sim = Sim::new();
        // The prefix: thirteen full blocks in one store.
        let prefix: Vec<i64> = (1..=13).collect();
        sim.deliver(&batch(1, None, vec![run(None, &prefix, 256)]));
        assert_eq!(sim.indexer.worker_block_count(sim.worker), 13);
        // A turn's decode crosses the next block boundary: the full block,
        // then the prompt's eight-token tail entry under the intermediate
        // hash one hash-block before it.
        sim.deliver(&batch(
            2,
            None,
            vec![run(Some(13), &[14], 256), run(Some(9001), &[901], 8)],
        ));
        // The next turn: the same full block again (a second copy) and its
        // own tail entry.
        sim.deliver(&batch(
            3,
            None,
            vec![run(Some(13), &[14], 256), run(Some(9002), &[902], 8)],
        ));
        assert_eq!(
            sim.state.members[0].index.counters.parentless_stores, 0,
            "a tail entry is not a parent-less store"
        );
        assert_eq!(sim.state.members[0].index.counters.parentless_blocks, 0);
        assert_eq!(
            (
                sim.state.members[0].index.counters.sub_block_stores,
                sim.state.members[0].index.counters.sub_block_blocks
            ),
            (2, 2),
            "the two entries are counted as sub-block stores"
        );
        assert_eq!(
            sim.indexer.worker_block_count(sim.worker),
            14,
            "the full blocks only"
        );
        // The eviction of each block lists the entry beside the full hash.
        sim.deliver(&batch(4, None, vec![removed(&[14, 901])]));
        assert_eq!(
            sim.indexer.worker_block_count(sim.worker),
            14,
            "the second copy of the full block stands"
        );
        sim.deliver(&batch(5, None, vec![removed(&[14, 902])]));
        assert_eq!(sim.indexer.worker_block_count(sim.worker), 13);
        sim.assert_matches_reference();
    }
}
