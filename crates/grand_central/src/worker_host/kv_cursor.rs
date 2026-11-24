//! The admission cursor per rank in front of the KV index (design 5.1, the
//! KV index row; 5.4, a standby subscribes itself).
//!
//! The cursor rules, per `(worker, rank)`: the first batch sets the cursor
//! wherever it is; a contiguous batch applies; a batch at or below the cursor
//! is a duplicate and is skipped, unless it carries a clear, sits a restart
//! window (1,024) or more below, is a counter at its start, or is the first
//! batch of a fresh connection below the cursor, which read as a restarted
//! publisher: the rank's blocks are cleared and the batch applied; a batch
//! past the cursor is a gap answered once by a replay request from the
//! expected sequence; a second gap is lost history, applied with the rank
//! marked degraded (its state kept under 1,024 missed batches, cleared
//! past). A snapshot chunk is applied outside these rules and moves the
//! cursor to its stamp. Never-drop flow control is the subscriber's: past the
//! index lanes' depth it stops reading the rank's stream.
//!
//! The tables' unit is a row, one rank of an engine: an id names the row,
//! and a rank repeats its rank as a check. The collector's unit is an
//! engine. What is an engine's (a snapshot's start, a new connection) the
//! collector says to each row of the engine, the rows a snapshot holds
//! nothing for included. A snapshot's start is a resync for
//! [`ResyncReason::Snapshot`] before its first block, and a chunk only adds.
//!
//! The facts that cross a batch's admission, its outcome, a reason and the
//! cursor a subscriber reads are the worker crate's port types
//! ([`smg_worker::kv_events::port`]): the tables state none of their own.
//!
//! [`ResyncReason::Snapshot`]: smg_worker::kv_events::port::ResyncReason::Snapshot

use smg_worker::kv_events::port::KvCursor;

/// A sequence this far below the cursor is a restarted publisher, not a late
/// duplicate: replays start at the sequence asked for, so a legitimate
/// duplicate sits just below the cursor.
pub const RESTART_WINDOW: u64 = 1_024;

/// Missed batches beyond this clear the rank instead of keeping its state:
/// the servicers' own decision when their replay buffer overflows.
pub const GAP_CLEAR_THRESHOLD: u64 = 1_024;

/// What the cursor decided about one batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchAdmission {
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

/// The admission cursor of one rank: the rules of this module's doc as one
/// state machine.
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
    pub fn admit(&mut self, seq: u64, clears: bool) -> BatchAdmission {
        let fresh = std::mem::replace(&mut self.fresh, false);
        let Some(last) = self.applied else {
            self.applied = Some(seq);
            self.replay = None;
            return BatchAdmission::Apply;
        };
        if seq == last + 1 {
            self.applied = Some(seq);
            self.replay = None;
            return BatchAdmission::Apply;
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
                return BatchAdmission::Stale;
            }
            self.applied = Some(seq);
            self.replay = None;
            self.degraded = false;
            return BatchAdmission::Restart;
        }
        let expected = last + 1;
        if self.replay.is_none() {
            self.replay = Some((expected, seq));
            return BatchAdmission::Replay { expected };
        }
        let missed = seq - expected;
        let cleared = missed > GAP_CLEAR_THRESHOLD;
        self.replay = None;
        self.degraded = !cleared;
        self.applied = Some(seq);
        BatchAdmission::Unrecovered { missed, cleared }
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

    /// The cursor as a subscriber reads it.
    pub fn view(&self) -> KvCursor {
        KvCursor {
            applied: self.applied.unwrap_or(0),
            replay_pending: self.replay.is_some(),
            degraded: self.degraded,
        }
    }
}
