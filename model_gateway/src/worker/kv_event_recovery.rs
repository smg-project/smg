//! Admission state for one KV-event stream rank: cursors, gap recovery and
//! the live tail held during an out-of-band resync.
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
//! stream is ordered.
//!
//! The tail buffer serves a snapshot resync delivered on a side channel: live
//! batches arriving while the snapshot is in flight are held, bounded, and
//! applied in order after it. No servicer sends one out of band today, so the
//! monitor never enters that mode; the logic is here, tested, for the
//! protocol in `crates/kv_index/docs/recovery-protocol.md`.

use std::collections::VecDeque;

use smg_grpc_client::common_proto::KvEventBatch;

/// A sequence this far below the cursor is a restarted publisher, not a late
/// duplicate: replays start at the sequence we asked for, so legitimate
/// duplicates sit just below the cursor.
pub(crate) const RESTART_WINDOW: u64 = 1_024;

/// Missed batches beyond this clear the rank instead of keeping its state:
/// the same decision the servicers make when their replay buffer overflows.
const GAP_CLEAR_THRESHOLD: u64 = 1_024;

/// Live batches held while a snapshot resync is in flight.
const TAIL_LIMIT: usize = 1_024;

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
    /// A snapshot resync is in flight: queue the batch in the tail.
    Buffered,
}

/// How the rank got where it is, for metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResyncReason {
    OutOfRange,
    DataLoss,
    PublisherRestart,
    GapCleared,
    /// The relay served a state snapshot in place of the history it no longer
    /// had: the worker's state is replaced by the live set it carries.
    Snapshot,
}

impl ResyncReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::OutOfRange => "out_of_range",
            Self::DataLoss => "data_loss",
            Self::PublisherRestart => "publisher_restart",
            Self::GapCleared => "gap_cleared",
            Self::Snapshot => "snapshot",
        }
    }
}

/// Admission state for one `(worker, dp_rank)`.
#[derive(Debug, Default)]
pub(crate) struct RankState {
    cursor: Cursor,
    replay: Option<PendingReplay>,
    /// Set when a gap could not be recovered; cleared by a resync.
    degraded: bool,
    /// Live batches held during a snapshot resync, oldest first.
    tail: VecDeque<KvEventBatch>,
    snapshot_inflight: bool,
    tail_overflowed: bool,
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

    pub(crate) fn tail_len(&self) -> usize {
        self.tail.len()
    }

    /// Decide what to do with a batch carrying `seq`; `clears` says whether
    /// it carries the engine's own cache clear.
    pub(crate) fn admit(&mut self, seq: u64, clears: bool) -> Admission {
        let fresh = std::mem::replace(&mut self.fresh, false);
        if self.snapshot_inflight {
            return Admission::Buffered;
        }
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

    /// Hold a live batch while a snapshot is in flight. Returns `false` when
    /// the tail is full; the batch is dropped and the resync must be redone.
    pub(crate) fn buffer_live(&mut self, batch: KvEventBatch) -> bool {
        if self.tail.len() >= TAIL_LIMIT {
            self.tail_overflowed = true;
            return false;
        }
        self.tail.push_back(batch);
        true
    }

    /// Enter snapshot mode: live batches are buffered until
    /// [`finish_snapshot`](Self::finish_snapshot).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "entered once a servicer serves snapshots out of band (docs/recovery-protocol.md)"
        )
    )]
    pub(crate) fn begin_snapshot(&mut self) {
        self.snapshot_inflight = true;
        self.tail.clear();
        self.tail_overflowed = false;
        self.replay = None;
    }

    /// The snapshot (complete through `through_seq`) has been applied. Returns
    /// the buffered live batches that come after it, in order and without
    /// duplicates, and sets the cursor to the last of them. `None` means the
    /// tail overflowed while waiting and the snapshot has to be taken again.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "entered once a servicer serves snapshots out of band (docs/recovery-protocol.md)"
        )
    )]
    pub(crate) fn finish_snapshot(&mut self, through_seq: u64) -> Option<Vec<KvEventBatch>> {
        self.snapshot_inflight = false;
        self.degraded = false;
        if self.tail_overflowed {
            self.tail.clear();
            self.tail_overflowed = false;
            self.cursor = Cursor::Live(through_seq);
            return None;
        }
        let mut cursor = through_seq;
        let mut out = Vec::new();
        for batch in self.tail.drain(..) {
            if batch.sequence_number <= cursor {
                continue;
            }
            cursor = batch.sequence_number;
            out.push(batch);
        }
        self.cursor = Cursor::Live(cursor);
        Some(out)
    }

    /// A snapshot chunk stamped `seq` was applied: the cursor stands there,
    /// whatever it was, and nothing is pending or degraded any more.
    pub(crate) fn resync_to(&mut self, seq: u64) {
        self.cursor = Cursor::Live(seq);
        self.replay = None;
        self.degraded = false;
        self.fresh = false;
        self.tail.clear();
        self.snapshot_inflight = false;
        self.tail_overflowed = false;
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
        self.tail.clear();
        self.snapshot_inflight = false;
        self.tail_overflowed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(seq: u64) -> KvEventBatch {
        KvEventBatch {
            sequence_number: seq,
            timestamp: 0.0,
            events: vec![],
            dp_rank: None,
            snapshot: None,
            load: None,
        }
    }

    #[test]
    fn first_batch_sets_the_cursor_wherever_it_is() {
        let mut rank = RankState::default();
        assert_eq!(rank.admit(17, false), Admission::Apply);
        assert_eq!(rank.cursor(), Cursor::Live(17));
        assert_eq!(rank.resume_from(), 17);
    }

    #[test]
    fn contiguous_batches_apply_and_duplicates_are_stale() {
        let mut rank = RankState::default();
        for seq in 1..=5 {
            assert_eq!(rank.admit(seq, false), Admission::Apply);
        }
        assert_eq!(rank.admit(4, false), Admission::Stale);
        assert_eq!(rank.admit(5, false), Admission::Stale);
        assert_eq!(rank.admit(6, false), Admission::Apply);
        assert_eq!(rank.cursor(), Cursor::Live(6));
    }

    /// A servicer that pushes load records sends `load_only` heartbeats
    /// with no events and the last sequence it sent. A gateway that predates
    /// the field (prost drops it) sees a repeat of its cursor: a late
    /// duplicate, dropped. It is never a restart (that needs a clear, a
    /// fresh connection below the cursor, a count back at 0 or 1, or a
    /// sequence a window below) and never a gap, however many arrive.
    #[test]
    fn a_repeat_of_the_cursor_is_stale_never_a_restart_or_a_gap() {
        let mut rank = RankState::default();
        for seq in 0..=7 {
            assert_eq!(rank.admit(seq, false), Admission::Apply);
        }
        for _ in 0..50 {
            assert_eq!(rank.admit(7, false), Admission::Stale);
        }
        assert_eq!(rank.cursor(), Cursor::Live(7));
        assert!(rank.replay_pending().is_none());
        assert_eq!(rank.admit(8, false), Admission::Apply);
        // After a reconnect too: a repeat of the cursor is not "below" it.
        rank.reconnected();
        assert_eq!(rank.admit(8, false), Admission::Stale);
        assert_eq!(rank.admit(9, false), Admission::Apply);
    }

    #[test]
    fn a_gap_asks_for_one_replay_then_resumes_when_the_server_fills_it() {
        let mut rank = RankState::default();
        for seq in 1..=5 {
            rank.admit(seq, false);
        }
        assert_eq!(rank.admit(8, false), Admission::Replay { expected: 6 });
        assert_eq!(rank.resume_from(), 5);
        assert!(rank.replay_pending().is_some());
        // The server resends from 6: everything is contiguous again.
        assert_eq!(rank.admit(6, false), Admission::Apply);
        assert!(rank.replay_pending().is_none());
        assert_eq!(rank.admit(7, false), Admission::Apply);
        assert_eq!(rank.admit(8, false), Admission::Apply);
        assert!(!rank.is_degraded());
    }

    #[test]
    fn a_gap_the_server_does_not_fill_keeps_state_and_marks_the_rank_degraded() {
        let mut rank = RankState::default();
        for seq in 1..=5 {
            rank.admit(seq, false);
        }
        assert_eq!(rank.admit(9, false), Admission::Replay { expected: 6 });
        // Reconnected from 6, but the server streams live from 9 again.
        assert_eq!(
            rank.admit(9, false),
            Admission::Unrecovered {
                missed: 3,
                cleared: false
            }
        );
        assert!(rank.is_degraded());
        assert_eq!(rank.cursor(), Cursor::Live(9));
        assert_eq!(rank.admit(10, false), Admission::Apply);
        // Never loops: the next gap starts a new, single replay attempt.
        assert_eq!(rank.admit(12, false), Admission::Replay { expected: 11 });
    }

    #[test]
    fn a_large_unfilled_gap_clears_the_rank() {
        let mut rank = RankState::default();
        rank.admit(1, false);
        let far = 2 + GAP_CLEAR_THRESHOLD + 1;
        assert_eq!(rank.admit(far, false), Admission::Replay { expected: 2 });
        assert_eq!(
            rank.admit(far, false),
            Admission::Unrecovered {
                missed: far - 2,
                cleared: true
            }
        );
        assert!(!rank.is_degraded());
        assert_eq!(rank.cursor(), Cursor::Live(far));
    }

    #[test]
    fn a_sequence_far_below_the_cursor_is_a_publisher_restart() {
        let mut rank = RankState::default();
        for seq in 1..=3_000 {
            rank.admit(seq, false);
        }
        assert_eq!(rank.admit(2_999, false), Admission::Stale);
        assert_eq!(
            rank.admit(3_000 - RESTART_WINDOW + 1, false),
            Admission::Stale
        );
        assert_eq!(
            rank.admit(3_000 - RESTART_WINDOW, false),
            Admission::Restart
        );
        assert_eq!(rank.cursor(), Cursor::Live(3_000 - RESTART_WINDOW));
        let mut fresh = RankState::default();
        for seq in 1..=3_000 {
            fresh.admit(seq, false);
        }
        assert_eq!(fresh.admit(1, false), Admission::Restart);
        assert_eq!(fresh.admit(2, false), Admission::Apply);
    }

    #[test]
    fn a_lower_sequence_on_a_fresh_connection_is_a_restart() {
        let mut rank = RankState::default();
        for seq in 1..=5 {
            rank.admit(seq, false);
        }
        rank.reconnected();
        assert_eq!(rank.admit(1, false), Admission::Restart);
        assert_eq!(rank.admit(2, false), Admission::Apply);
        assert_eq!(rank.admit(3, false), Admission::Apply);
        // Mid-stream, a sequence just below the cursor is a duplicate.
        assert_eq!(rank.admit(2, false), Admission::Stale);
        // A replay that starts exactly at the cursor is a duplicate too.
        rank.reconnected();
        assert_eq!(rank.admit(3, false), Admission::Stale);
        assert_eq!(rank.admit(4, false), Admission::Apply);
        // A gap on a fresh connection is still a gap.
        rank.reconnected();
        assert_eq!(rank.admit(9, false), Admission::Replay { expected: 5 });
    }

    #[test]
    fn a_counter_at_its_start_below_the_cursor_is_a_restart() {
        let mut rank = RankState::default();
        for seq in 1..=5 {
            rank.admit(seq, false);
        }
        // vLLM and SGLang count from 0, the mock engine from 1; neither
        // number can be a replayed duplicate under a cursor of 5.
        assert_eq!(rank.admit(0, false), Admission::Restart);
        assert_eq!(rank.admit(1, false), Admission::Apply);
        for seq in 2..=5 {
            rank.admit(seq, false);
        }
        assert_eq!(rank.admit(1, false), Admission::Restart);
        assert_eq!(rank.cursor(), Cursor::Live(1));
    }

    #[test]
    fn a_clear_below_the_cursor_is_a_restart() {
        let mut rank = RankState::default();
        for seq in 1..=5 {
            rank.admit(seq, false);
        }
        assert_eq!(rank.admit(0, true), Admission::Restart);
        assert_eq!(rank.cursor(), Cursor::Live(0));
        assert_eq!(rank.admit(1, false), Admission::Apply);
        // A clear at or above the cursor is an ordinary event.
        assert_eq!(rank.admit(2, true), Admission::Apply);
    }

    #[test]
    fn a_snapshot_chunk_moves_the_cursor_wherever_it_is_stamped() {
        let mut rank = RankState::default();
        for seq in 1..=5 {
            rank.admit(seq, false);
        }
        assert_eq!(rank.admit(9, false), Admission::Replay { expected: 6 });
        rank.resync_to(40);
        assert_eq!(rank.cursor(), Cursor::Live(40));
        assert!(rank.replay_pending().is_none());
        assert_eq!(rank.admit(41, false), Admission::Apply);
        // Below the cursor as well: a stale cursor is replaced, not restarted.
        rank.reconnected();
        rank.resync_to(3);
        assert_eq!(rank.admit(4, false), Admission::Apply);
        assert_eq!(rank.resume_from(), 4);
    }

    #[test]
    fn reset_forgets_the_cursor() {
        let mut rank = RankState::default();
        rank.admit(40, false);
        rank.admit(42, false);
        rank.reset();
        assert_eq!(rank.cursor(), Cursor::Initial);
        assert_eq!(rank.resume_from(), 0);
        assert_eq!(rank.admit(7, false), Admission::Apply);
    }

    #[test]
    fn snapshot_mode_buffers_live_batches_and_replays_the_tail_in_order() {
        let mut rank = RankState::default();
        for seq in 1..=5 {
            rank.admit(seq, false);
        }
        rank.begin_snapshot();
        assert_eq!(rank.admit(6, false), Admission::Buffered);
        assert!(rank.buffer_live(batch(6)));
        assert!(rank.buffer_live(batch(7)));
        assert!(rank.buffer_live(batch(7)));
        assert!(rank.buffer_live(batch(8)));
        assert_eq!(rank.tail_len(), 4);
        // The snapshot covered everything through 6.
        let tail = rank.finish_snapshot(6).expect("tail intact");
        let seqs: Vec<u64> = tail.iter().map(|b| b.sequence_number).collect();
        assert_eq!(seqs, vec![7, 8]);
        assert_eq!(rank.cursor(), Cursor::Live(8));
        assert_eq!(rank.admit(9, false), Admission::Apply);
        assert!(!rank.is_degraded());
    }

    #[test]
    fn a_tail_overflow_is_bounded_and_forces_another_snapshot() {
        let mut rank = RankState::default();
        rank.admit(1, false);
        rank.begin_snapshot();
        for seq in 2..=(TAIL_LIMIT as u64 + 1) {
            assert!(rank.buffer_live(batch(seq)));
        }
        assert_eq!(rank.tail_len(), TAIL_LIMIT);
        assert!(!rank.buffer_live(batch(TAIL_LIMIT as u64 + 2)));
        assert_eq!(rank.tail_len(), TAIL_LIMIT);
        assert!(rank.finish_snapshot(1).is_none());
        assert_eq!(rank.tail_len(), 0);
        assert_eq!(rank.cursor(), Cursor::Live(1));
    }
}
