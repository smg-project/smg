//! Admission state for one KV-event stream rank: cursors and gap recovery.
//!
//! Every engine publisher (one per data-parallel rank) numbers its batches
//! with its own monotonic sequence. The subscriber keeps one [`RankState`] per
//! `(worker, dp_rank)` and asks it what to do with each batch it receives:
//!
//! - contiguous and first batches are applied;
//! - duplicates (a replay overlapping what was already applied) are skipped;
//! - a publisher restart clears the rank's state and applies the batch: the
//!   engine's cache is empty again. It shows as a sequence below the cursor on
//!   a fresh connection (servers resume strictly after the cursor, so nothing
//!   legitimate sits there), a cache clear carried below the cursor (SGLang's
//!   first batch after startup), a counter at its start (0 or 1) below the
//!   cursor, or a sequence far below the cursor mid-stream;
//! - a gap is answered once by asking the server to replay from the expected
//!   sequence; if the next batch still skips ahead the server kept no history
//!   (the Rust relay never replays, the SGLang servicer only within its
//!   buffer) and the rank is marked degraded: a small gap keeps the existing
//!   state (an engine still holds most of those blocks and will never re-send
//!   stores for them), a large one clears it, mirroring the servicers' own
//!   `OUT_OF_RANGE` / `DATA_LOSS` signal.
//!
//! The Rust relay serves a state snapshot in band: once its history no longer
//! starts at the publisher's first batch, a subscription from zero begins with
//! chunks marked `KvSnapshotChunk` (chunk 0 carries the engine's clear, the
//! stores follow parents first) stamped with consecutive sequence numbers
//! ending at the sequence the state was cut at, and live events continue
//! from the next one. The monitor applies the chunks as a `snapshot` resync
//! outside the admission rules above and sets the rank's cursor to each
//! chunk's stamp ([`RankState::resync_to`]); nothing is buffered because the
//! stream is ordered. A snapshot delivered on a side channel, with live
//! batches held meanwhile, is not implemented: no servicer serves one, and
//! this module carries nothing for it until one does.

/// A sequence this far below the cursor is a restarted publisher, not a late
/// duplicate: replays start at the sequence we asked for, so legitimate
/// duplicates sit just below the cursor.
pub(crate) const RESTART_WINDOW: u64 = 1_024;

/// Missed batches beyond this clear the rank instead of keeping its state:
/// the same decision the servicers make when their replay buffer overflows.
pub(crate) const GAP_CLEAR_THRESHOLD: u64 = 1_024;

/// Where the rank's cursor stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Cursor {
    /// Nothing admitted yet: the first batch sets the cursor wherever it is.
    #[default]
    Initial,
    /// The last sequence number applied.
    Live(u64),
}

/// A replay the rank asked the server for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingReplay {
    /// The sequence the server was asked to resume from.
    expected: u64,
    /// The sequence that revealed the gap.
    received: u64,
}

/// What the subscriber must do with a batch.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Admission {
    /// Apply the batch's events; the cursor advanced to it.
    Apply,
    /// A duplicate of something already applied; drop it.
    Stale,
    /// The publisher restarted: clear the rank's blocks, then apply.
    Restart,
    /// A gap with no replay asked yet: reconnect with `expected` as the start
    /// sequence; the batch itself is not applied (the replay will resend it).
    Replay { expected: u64 },
    /// A gap the server could not fill: `missed` batches are lost. The rank's
    /// state is kept (`cleared == false`) or dropped (`cleared == true`); then
    /// apply the batch.
    Unrecovered { missed: u64, cleared: bool },
}

/// Admission state for one `(worker, dp_rank)`.
#[derive(Debug, Default)]
pub(crate) struct RankState {
    cursor: Cursor,
    replay: Option<PendingReplay>,
    /// Set when a gap could not be recovered; cleared by a resync.
    degraded: bool,
    /// No batch received yet on the current connection.
    fresh: bool,
}

impl RankState {
    #[cfg(test)]
    pub(crate) fn cursor(&self) -> Cursor {
        self.cursor
    }

    /// The cursor to send when (re)subscribing: the last sequence applied, or
    /// 0 for none. Servers resume strictly after it (the SGLang servicer asks
    /// its engine for `cursor + 1`, the mock engine streams `> cursor`), so a
    /// pending replay needs nothing more: its `expected` is `cursor + 1`.
    pub(crate) fn resume_from(&self) -> u64 {
        match self.cursor {
            Cursor::Live(last) => last,
            Cursor::Initial => 0,
        }
    }

    /// A new stream connection was made. The first batch on it shows where
    /// the server stands: below the cursor there means a restarted publisher,
    /// not a late duplicate.
    pub(crate) fn reconnected(&mut self) {
        self.fresh = true;
    }

    pub(crate) fn is_degraded(&self) -> bool {
        self.degraded
    }

    #[cfg(test)]
    pub(crate) fn replay_pending(&self) -> Option<PendingReplay> {
        self.replay
    }

    /// The batches a pending replay was asked for (`received - expected`),
    /// 0 when none is pending. Read before a reset drops the replay, so a
    /// resume the server refused still counts what it skipped.
    pub(crate) fn pending_gap(&self) -> u64 {
        self.replay
            .map_or(0, |replay| replay.received - replay.expected)
    }

    /// Decide what to do with a batch carrying `seq`; `clears` says whether
    /// it carries the engine's own cache clear.
    pub(crate) fn admit(&mut self, seq: u64, clears: bool) -> Admission {
        let fresh = std::mem::replace(&mut self.fresh, false);
        match self.cursor {
            Cursor::Initial => {
                self.cursor = Cursor::Live(seq);
                self.replay = None;
                Admission::Apply
            }
            Cursor::Live(last) if seq == last + 1 => {
                self.cursor = Cursor::Live(seq);
                self.replay = None;
                Admission::Apply
            }
            Cursor::Live(last) if seq <= last => {
                // A counter at its start is a new publisher too: replays
                // never reach below `cursor + 1`, so 0 or 1 under a cursor of
                // 2 or more cannot be a late duplicate.
                let restarted = clears
                    || (fresh && seq < last)
                    || (seq <= 1 && seq < last)
                    || seq + RESTART_WINDOW <= last;
                if restarted {
                    // A fresh publisher counting from the start again.
                    self.cursor = Cursor::Live(seq);
                    self.replay = None;
                    self.degraded = false;
                    Admission::Restart
                } else {
                    Admission::Stale
                }
            }
            Cursor::Live(last) => {
                let expected = last + 1;
                match self.replay {
                    None => {
                        self.replay = Some(PendingReplay {
                            expected,
                            received: seq,
                        });
                        Admission::Replay { expected }
                    }
                    Some(pending) => {
                        // We already asked to resume from `expected` and the
                        // server skipped ahead anyway: no history there.
                        debug_assert_eq!(pending.expected, expected);
                        let missed = seq - expected;
                        let cleared = missed > GAP_CLEAR_THRESHOLD;
                        self.replay = None;
                        self.degraded = !cleared;
                        self.cursor = Cursor::Live(seq);
                        Admission::Unrecovered { missed, cleared }
                    }
                }
            }
        }
    }

    /// A snapshot chunk stamped `seq` was applied: the cursor stands there,
    /// whatever it was, and nothing is pending or degraded any more.
    pub(crate) fn resync_to(&mut self, seq: u64) {
        self.cursor = Cursor::Live(seq);
        self.replay = None;
        self.degraded = false;
        self.fresh = false;
    }

    /// The relay's snapshot could not cover the engine's whole life
    /// (`KvSnapshotChunk.unknown_before`): the rank's blocks are a partial
    /// set until the next resync, as after an unrecovered gap.
    pub(crate) fn mark_degraded(&mut self) {
        self.degraded = true;
    }

    /// The server declared its history gone (`OUT_OF_RANGE` / `DATA_LOSS`) or
    /// the subscriber decided to drop the rank: forget the cursor so the next
    /// stream is taken from wherever it starts.
    pub(crate) fn reset(&mut self) {
        self.cursor = Cursor::Initial;
        self.replay = None;
        self.degraded = false;
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/policies/kv_index/event_monitor/recovery/tests.rs"]
mod tests;
