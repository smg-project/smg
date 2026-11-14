//! The KV event sink: the admission cursor per rank in front of the index
//! (design 5.1, the KV index row; 5.4, a standby subscribes itself).

use crate::view::WorkerId;

/// The header facts of one batch, read by the cursor; the batch itself (the
/// servicer's message, with the blocks the index applies) rides beside them
/// as the host's own type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchFacts {
    /// The publisher's sequence number: monotonic per rank, each rank its
    /// own counter.
    pub sequence: u64,
    /// The batch carries the engine's own cache clear.
    pub clears: bool,
    /// The batch is one chunk of a relay's state snapshot, not a batch the
    /// publisher sent.
    pub snapshot: Option<SnapshotChunk>,
}

/// Where a relay's in-band snapshot stands: chunk `index` of `count`, each
/// stamped with a sequence the relay chose so live batches continue after
/// the last one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SnapshotChunk {
    pub index: u32,
    pub count: u32,
    /// The snapshot could not cover the engine's whole life: the rank's
    /// blocks are a partial set until the next resync.
    pub unknown_before: bool,
}

/// What the sink did with a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchOutcome {
    /// Applied to the index; the cursor advanced to it.
    Applied,
    /// A duplicate below the cursor, a chunk of a snapshot still arriving, or
    /// another rank's batch: counted, not applied.
    Skipped,
    /// A gap with no replay asked yet: the caller reconnects asking for a
    /// replay from `expected`; the batch is not applied (the replay resends
    /// it). A second gap on the same rank is applied as lost history (see
    /// [`KvCursor::degraded`]).
    Gap { expected: u64, received: u64 },
}

/// Why a rank's cursor was reset or its blocks replaced. The first six are
/// the servicer's own stream status codes, forwarded by a relay as frames so
/// the store tells a relay restart from an engine restart; the last three
/// the cursor's own findings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResyncReason {
    /// The server's history no longer reaches the cursor.
    OutOfRange,
    /// The server lost batches it cannot name.
    DataLoss,
    StreamEnded,
    /// The stream ended with a snapshot still arriving: a partial live set.
    SnapshotAbandoned,
    RelayRestart,
    EngineReplaced,
    /// A sequence far below the cursor, or a clear below it: the publisher
    /// started over and its cache is empty.
    PublisherRestart,
    /// A gap the server could not replay, too large to keep the state.
    GapCleared,
    /// A relay served a state snapshot in place of history: the live set it
    /// carries replaces the rank's blocks.
    Snapshot,
}

impl ResyncReason {
    /// The metric label of the reason.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutOfRange => "out_of_range",
            Self::DataLoss => "data_loss",
            Self::StreamEnded => "stream_ended",
            Self::SnapshotAbandoned => "snapshot_abandoned",
            Self::RelayRestart => "relay_restart",
            Self::EngineReplaced => "engine_replaced",
            Self::PublisherRestart => "publisher_restart",
            Self::GapCleared => "gap_cleared",
            Self::Snapshot => "snapshot",
        }
    }
}

/// One rank's cursor as a subscriber reads it before it (re)subscribes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvCursor {
    /// The last sequence applied, `0` for none: servers resume strictly
    /// after it.
    pub applied: u64,
    /// A replay was asked for and its batches have not arrived.
    pub replay_pending: bool,
    /// A gap could not be recovered: the rank's blocks are a partial set
    /// until a resync.
    pub degraded: bool,
}

/// Where KV event batches are admitted and applied.
///
/// The cursor rules, per `(worker, rank)`: the first batch sets the cursor
/// wherever it is; a contiguous batch applies; a batch at or below the cursor
/// is a duplicate and is skipped, unless it carries a clear, sits a restart
/// window (1,024) or more below, is a counter at its start, or is the first
/// batch of a fresh connection below the cursor, which read as a restarted
/// publisher: the rank's blocks are cleared and the batch applied; a batch
/// past the cursor is a gap answered once by a replay request from the
/// expected sequence; a second gap is lost history, applied with the rank
/// marked degraded (its state kept under 1,024 missed batches, cleared
/// past). A snapshot chunk is applied outside these rules and moves the
/// cursor to its stamp. Never-drop flow control is the subscriber's: past the
/// index lanes' depth it stops reading the rank's stream.
pub trait KvEventSink: Send + Sync {
    /// The batch message as the host's wire crate defines it; the store
    /// applies its events to the index and reads nothing else of it.
    type Batch: ?Sized;

    /// Run one batch of `rank` through its cursor and, when admitted, apply
    /// it to the worker's blocks. `facts` are the batch's header as the
    /// cursor reads it. A load-only batch carries no events and is not
    /// admitted: its record goes to the load sink.
    fn admit(
        &self,
        id: WorkerId,
        rank: u16,
        facts: BatchFacts,
        batch: &Self::Batch,
    ) -> BatchOutcome;

    /// A new stream connection was made for the worker: the first batch on
    /// it shows where the publisher stands.
    fn reconnected(&self, id: WorkerId);

    /// Reset one rank's cursor, or every rank's (`None`), for `reason`; the
    /// worker's blocks go with it, and the next stream is taken from
    /// wherever it starts.
    fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason);

    /// The cursor of one rank, when the rank has been heard from.
    fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor>;

    /// The worker left: its blocks out of the index, its cursors gone. A
    /// same-key successor is interned only after this returns.
    fn removed(&self, id: WorkerId);

    /// The block size of record for a model, once a stored block or a
    /// registration named it.
    fn block_size(&self, model: &str) -> Option<u32>;

    /// Name a model's block size: provisional until a stored block confirms
    /// it, and only ever raised.
    fn set_block_size(&self, model: &str, size: u32);
}

/// A sequence this far below the cursor is a restarted publisher, not a late
/// duplicate: replays start at the sequence asked for, so a legitimate
/// duplicate sits just below the cursor.
pub const RESTART_WINDOW: u64 = 1_024;

/// Missed batches beyond this clear the rank instead of keeping its state:
/// the servicers' own decision when their replay buffer overflows.
pub const GAP_CLEAR_THRESHOLD: u64 = 1_024;

/// What the cursor decided about one batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Apply the batch; the cursor advanced to it.
    Apply,
    /// A duplicate of something already applied; drop it.
    Stale,
    /// The publisher restarted: clear the rank's blocks, then apply.
    Restart,
    /// A gap with no replay asked yet: reconnect from `expected`; the batch
    /// itself is not applied (the replay resends it).
    Replay { expected: u64 },
    /// A gap the server could not fill: `missed` batches are lost; the
    /// rank's state is kept or `cleared`; then apply.
    Unrecovered { missed: u64, cleared: bool },
}

impl Admission {
    /// The batch is applied after this decision.
    pub const fn applies(self) -> bool {
        matches!(self, Self::Apply | Self::Restart | Self::Unrecovered { .. })
    }

    /// The rank's blocks are cleared before the batch is applied.
    pub const fn clears_rank(self) -> bool {
        matches!(
            self,
            Self::Restart | Self::Unrecovered { cleared: true, .. }
        )
    }
}

/// The admission cursor of one rank: the rules of [`KvEventSink`] as one
/// state machine, shared by every store (the one-process store, a gateway
/// host, the controller's moved monitor).
#[derive(Clone, Copy, Debug, Default)]
/// The gateway's event monitor keeps its own copy of these rules until the
/// step that moves the ingestion drivers points it at this cursor; the two
/// stay in lockstep until then.
pub struct RankCursor {
    /// The last sequence applied; `None` before the first batch.
    applied: Option<u64>,
    /// A replay asked for: the expected sequence and the one that revealed
    /// the gap.
    replay: Option<(u64, u64)>,
    degraded: bool,
    /// No batch yet on the current connection.
    fresh: bool,
}

impl RankCursor {
    /// Decide what to do with a batch carrying `seq`; `clears` says whether
    /// it carries the engine's own cache clear.
    pub fn admit(&mut self, seq: u64, clears: bool) -> Admission {
        let fresh = std::mem::replace(&mut self.fresh, false);
        let Some(last) = self.applied else {
            self.applied = Some(seq);
            self.replay = None;
            return Admission::Apply;
        };
        if seq == last + 1 {
            self.applied = Some(seq);
            self.replay = None;
            return Admission::Apply;
        }
        if seq <= last {
            // A counter at its start is a new publisher too: replays never
            // reach below the cursor plus one, so 0 or 1 under a cursor of
            // 2 or more cannot be a late duplicate.
            let restarted = clears
                || (fresh && seq < last)
                || (seq <= 1 && seq < last)
                || seq + RESTART_WINDOW <= last;
            if !restarted {
                return Admission::Stale;
            }
            self.applied = Some(seq);
            self.replay = None;
            self.degraded = false;
            return Admission::Restart;
        }
        let expected = last + 1;
        if self.replay.is_none() {
            self.replay = Some((expected, seq));
            return Admission::Replay { expected };
        }
        let missed = seq - expected;
        let cleared = missed > GAP_CLEAR_THRESHOLD;
        self.replay = None;
        self.degraded = !cleared;
        self.applied = Some(seq);
        Admission::Unrecovered { missed, cleared }
    }

    /// A snapshot chunk stamped `seq` was applied: the cursor stands there,
    /// nothing is pending, and the rank is `partial` when the snapshot could
    /// not cover the engine's whole life.
    pub fn resync_to(&mut self, seq: u64, partial: bool) {
        self.applied = Some(seq);
        self.replay = None;
        self.degraded = partial;
        self.fresh = false;
    }

    /// A new stream connection was made: the first batch on it shows where
    /// the publisher stands.
    pub fn reconnected(&mut self) {
        self.fresh = true;
    }

    /// Forget the cursor: the next stream is taken from wherever it starts.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The batches a pending replay was asked for; `0` when none is pending.
    pub fn pending_gap(&self) -> u64 {
        self.replay
            .map_or(0, |(expected, received)| received - expected)
    }

    /// The cursor as a subscriber reads it.
    pub fn view(&self) -> KvCursor {
        KvCursor {
            applied: self.applied.unwrap_or(0),
            replay_pending: self.replay.is_some(),
            degraded: self.degraded,
        }
    }
}
