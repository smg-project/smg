use super::*;

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
