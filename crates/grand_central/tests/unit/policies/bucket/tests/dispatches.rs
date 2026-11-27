//! What a dispatch books: the worker that serves, the booking's date
//! against the window, and the two sides of a pair.

use super::*;

/// A lease's first dispatch as a served host tells it: the lease names
/// the target the dispatch took by its `index` in the pick, and the
/// target names its worker and no position.
fn dispatch(
    scorer: &Bucket,
    worker: WorkerId,
    index: u8,
    tokens: u32,
    issued_micros: u64,
    since_micros: u32,
) {
    let lease = Lease {
        target_index: index,
        tokens,
        issued_micros,
        ..Lease::default()
    };
    let target = Target {
        id: worker,
        ..Target::default()
    };
    scorer.on_dispatched(&lease, &target, Side::Single, since_micros);
}

/// The date of the booking model `m` took last.
fn last_booked_at(scorer: &Bucket) -> Option<u64> {
    let partition = of(scorer, "m");
    let pool = partition.lock();
    pool.window.back().map(|booking| booking.at_micros)
}

#[test]
fn a_first_dispatch_on_another_target_books_the_worker_that_serves() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    // A lease is issued on its pick's first target, the first row here.
    book(&scorer, &pool, 0, 300, SECOND);
    // A dispatch on that target finds its booking made.
    dispatch(&scorer, pool[0].id, 0, 300, SECOND, 250);
    assert_eq!(loads(&scorer), (vec![300, 0, 0], 300));
    // Another decision books meanwhile, and the lease's gateway
    // dispatches on the pick's second target, the third row: that
    // worker serves and is booked, at the dispatch's instant. The
    // booking of the issue stands: nothing is un-booked.
    book(&scorer, &pool, 1, 10, SECOND + 400);
    dispatch(&scorer, pool[2].id, 1, 300, SECOND, 250);
    assert_eq!(loads(&scorer), (vec![300, 10, 300], 610));
    assert_eq!(last_booked_at(&scorer), Some(SECOND + 250));
    // The booking of the issue ages off the first target with its
    // interval, and the dispatch's stands on.
    book(&scorer, &pool, 1, 1, 6 * SECOND + 1);
    assert_eq!(loads(&scorer), (vec![0, 11, 300], 311));
    // A worker no partition lists takes no booking.
    dispatch(&scorer, WorkerId(999), 1, 300, 6 * SECOND, 250);
    assert_eq!(loads(&scorer), (vec![0, 11, 300], 311));
}

#[test]
fn a_dispatch_s_booking_is_dated_no_later_than_the_newest_on_file() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(2);
    led(&scorer, &pool, 0, SECOND);
    // A partition with no booking on file takes the gateway's date.
    dispatch(&scorer, pool[1].id, 1, 50, SECOND, 250);
    assert_eq!(last_booked_at(&scorer), Some(SECOND + 250));
    // A date far past the newest booking is a gateway's word: taken
    // whole it would age every booking on file. The booking is dated
    // at the newest.
    book(&scorer, &pool, 0, 40, 2 * SECOND);
    dispatch(&scorer, pool[1].id, 1, 7, 2 * SECOND, u32::MAX);
    assert_eq!(loads(&scorer), (vec![40, 57], 97));
    assert_eq!(last_booked_at(&scorer), Some(2 * SECOND));
    // And no earlier than its lease's issue, where the partition
    // booked nothing since.
    dispatch(&scorer, pool[1].id, 1, 3, 3 * SECOND, 250);
    assert_eq!(loads(&scorer), (vec![40, 60], 100));
    assert_eq!(last_booked_at(&scorer), Some(3 * SECOND));
}

#[test]
fn a_dispatch_told_past_its_interval_books_nothing() {
    let scorer = scorer(NEVER, 1.5, 1);
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    // A lease is issued on the first row. Its booking has aged when
    // another decision books the second row two seconds later.
    book(&scorer, &pool, 0, 300, SECOND);
    book(&scorer, &pool, 1, 10, 3 * SECOND);
    assert_eq!(loads(&scorer), (vec![0, 10, 0], 10));
    // The first lease's dispatch on the third row is told then, dated
    // 250 microseconds behind its decision: more than an interval
    // before the booking on file. It aged before it was told, and books
    // nothing.
    dispatch(&scorer, pool[2].id, 1, 300, SECOND, 250);
    assert_eq!(loads(&scorer), (vec![0, 10, 0], 10));
    assert_eq!(last_booked_at(&scorer), Some(3 * SECOND));
}

#[test]
fn a_dispatch_s_booking_ends_no_booking_newer_than_itself() {
    let scorer = scorer(NEVER, 1.5, 1);
    let pool = pool(3);
    let tenth = SECOND / 10;
    led(&scorer, &pool, 0, 10 * SECOND);
    for (tokens, at_micros) in [(1, 100 * tenth), (2, 105 * tenth), (4, 107 * tenth)] {
        book(&scorer, &pool, 0, tokens, at_micros);
    }
    // A dispatch dated one interval before the oldest booking, to the
    // microsecond, arrived out of its clock's order: it is booked,
    // behind the three.
    dispatch(&scorer, pool[1].id, 1, 8, 90 * tenth, 0);
    assert_eq!(loads(&scorer), (vec![7, 8, 0], 15));
    assert_eq!(last_booked_at(&scorer), Some(90 * tenth));
    // A dispatch dated more than an interval before the oldest booking
    // is held against that one, not against the last one made: it books
    // nothing, and the three newer bookings stand.
    dispatch(&scorer, pool[2].id, 1, 16, 89 * tenth, 0);
    assert_eq!(loads(&scorer), (vec![7, 8, 0], 15));
}

#[test]
fn a_late_dispatch_into_a_full_window_ends_no_newer_booking() {
    let scorer = scorer(NEVER, 1.5, 1);
    let pool = pool(3);
    let tenth = SECOND / 10;
    led(&scorer, &pool, 0, 10 * SECOND);
    book(&scorer, &pool, 0, 1, 100 * tenth);
    for _ in 2..WINDOW_CAP {
        book(&scorer, &pool, 1, 1, 105 * tenth);
    }
    let full = WINDOW_CAP as u64;
    assert_eq!(loads(&scorer), (vec![1, full - 2, 0], full - 1));
    // One booking short of its cap the window loses no front unaged,
    // and the front alone holds the date: a dispatch dated within an
    // interval of it is booked, and fills the window.
    dispatch(&scorer, pool[2].id, 1, 8, 92 * tenth, 0);
    assert_eq!(loads(&scorer), (vec![1, full - 2, 8], full + 7));
    assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
    // A full window loses its front to the cap, whatever its age, and
    // the booking behind it is the first one aged. The same dispatch
    // now books nothing: more than an interval before that booking, it
    // aged before it was told.
    dispatch(&scorer, pool[2].id, 1, 8, 92 * tenth, 0);
    assert_eq!(loads(&scorer), (vec![1, full - 2, 8], full + 7));
    assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
    // One dated exactly an interval before that booking is booked: the
    // guard admits a date exactly an interval away, and the cap ends
    // the front alone.
    dispatch(&scorer, pool[2].id, 1, 8, 95 * tenth, 0);
    assert_eq!(loads(&scorer), (vec![0, full - 2, 16], full + 14));
    assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
}

#[test]
fn a_late_dispatch_is_held_against_the_front_of_a_full_window_too() {
    let scorer = scorer(NEVER, 1.5, 1);
    let pool = pool(3);
    let tenth = SECOND / 10;
    led(&scorer, &pool, 0, 10 * SECOND);
    // The second booking arrived out of its clock's order, an interval
    // before the first, and the window fills behind the two.
    book(&scorer, &pool, 0, 1, 100 * tenth);
    book(&scorer, &pool, 1, 1, 90 * tenth);
    for _ in 2..WINDOW_CAP {
        book(&scorer, &pool, 1, 1, 105 * tenth);
    }
    let full = WINDOW_CAP as u64;
    assert_eq!(loads(&scorer), (vec![1, full - 1, 0], full));
    // A dispatch dated within an interval of the booking behind the
    // front, and more than an interval before the front, aged before it
    // was told, as it did in a window short of its cap: it books
    // nothing.
    dispatch(&scorer, pool[2].id, 1, 8, 85 * tenth, 0);
    assert_eq!(loads(&scorer), (vec![1, full - 1, 0], full));
    assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
    // One dated within an interval of both is booked, and the cap ends
    // the front alone: the booking step stops at the one behind it,
    // and the newer bookings behind that one stand.
    dispatch(&scorer, pool[2].id, 1, 8, 92 * tenth, 0);
    assert_eq!(loads(&scorer), (vec![0, full - 1, 8], full + 7));
    assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
}

#[test]
fn a_late_dispatch_ends_no_booking_newer_than_its_date() {
    let scorer = scorer(NEVER, 1.5, 1);
    let pool = pool(3);
    let tenth = SECOND / 10;
    led(&scorer, &pool, 0, 10 * SECOND);
    // Three bookings arrive out of their clocks' order, and the window
    // fills behind them. The first of the newer ones ends the first of
    // the three as aged: the full window stands on a front, an older
    // booking behind it, and newer ones behind that.
    book(&scorer, &pool, 0, 1, 96 * tenth);
    book(&scorer, &pool, 0, 1, 105 * tenth);
    book(&scorer, &pool, 1, 1, 90 * tenth);
    for _ in 2..WINDOW_CAP {
        book(&scorer, &pool, 1, 1, 112 * tenth);
    }
    let full = WINDOW_CAP as u64;
    assert_eq!(loads(&scorer), (vec![1, full - 1, 0], full));
    assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
    // A dispatch dated no more than an interval before the front, and
    // no more than an interval before the booking behind it, is booked.
    // The cap ends the front, and the booking behind it leaves as aged,
    // more than an interval older than the dispatch. The newer ones
    // stand: the dispatch's date is late, and no clock that went back.
    dispatch(&scorer, pool[2].id, 1, 8, 101 * tenth, 0);
    assert_eq!(loads(&scorer), (vec![0, full - 2, 8], full + 6));
    assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP - 1);
}

#[test]
fn a_late_dispatch_ends_no_newer_booking_short_of_the_cap_either() {
    let scorer = scorer(NEVER, 1.5, 1);
    let pool = pool(3);
    let hundredth = SECOND / 100;
    led(&scorer, &pool, 0, 100 * SECOND);
    // Two issues an interval apart, and a dispatch dated before both,
    // within an interval of the front.
    book(&scorer, &pool, 0, 1, 10_020 * hundredth);
    book(&scorer, &pool, 0, 2, 10_120 * hundredth);
    dispatch(&scorer, pool[2].id, 1, 4, 9_920 * hundredth, 100_000);
    assert_eq!(loads(&scorer), (vec![3, 0, 4], 7));
    // An issue at 101.9 s ends the front as aged. One at 100.15 s, a
    // clock that stepped back, ends the next front. The dispatch's
    // booking stands at the front, the booking of 101.9 s behind it.
    book(&scorer, &pool, 1, 8, 10_190 * hundredth);
    book(&scorer, &pool, 1, 16, 10_015 * hundredth);
    assert_eq!(loads(&scorer), (vec![0, 24, 4], 28));
    // A dispatch dated 100.5 s: more than an interval after the front,
    // and more than an interval before the booking behind it. The front
    // leaves as aged. The newer booking stands, in a window far short
    // of its cap.
    dispatch(&scorer, pool[2].id, 1, 32, 10_050 * hundredth, 0);
    assert_eq!(loads(&scorer), (vec![0, 24, 32], 56));
    assert_eq!(of(&scorer, "m").lock().window.len(), 3);
}

#[test]
fn a_dispatch_books_each_partition_that_lists_its_worker() {
    let scorer = scorer(NEVER, 1.5, 5);
    let (pool_a, pool_b) = (pool(2), pool(4));
    for (model, pool) in [("a", &pool_a), ("b", &pool_b)] {
        led_by(&scorer, &prefill(model, 0, SECOND), pool, &all(pool));
    }
    // Both models list the second row, and one alone the fourth.
    dispatch(&scorer, pool_a[1].id, 1, 30, SECOND, 250);
    dispatch(&scorer, pool_b[3].id, 1, 9, SECOND, 250);
    let booked = |model: &str| {
        let partition = of(&scorer, model);
        let held = partition.lock();
        let loads: Vec<u64> = held.members.iter().map(|member| member.load).collect();
        (loads, held.total)
    };
    assert_eq!(booked("a"), (vec![0, 30], 30));
    assert_eq!(booked("b"), (vec![0, 30, 0, 9], 39));
}

#[test]
fn a_dispatch_on_another_target_of_the_issue_s_worker_books_nothing() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    book(&scorer, &pool, 0, 300, SECOND);
    // The lease its host tells names the worker its issue booked.
    let told = |index: u8, target: Target| {
        let lease = Lease {
            target_index: index,
            booked: pool[0].id,
            tokens: 300,
            issued_micros: SECOND,
            ..Lease::default()
        };
        scorer.on_dispatched(&lease, &target, Side::Single, 250);
    };
    // The pick's second target is another rank of that worker: the
    // booking of the issue is the serving worker's own.
    told(
        1,
        Target {
            id: pool[0].id,
            dp_rank: 1,
            ..Target::default()
        },
    );
    assert_eq!(loads(&scorer), (vec![300, 0, 0], 300));
    // Its third target is another worker, which is booked.
    told(
        2,
        Target {
            id: pool[2].id,
            ..Target::default()
        },
    );
    assert_eq!(loads(&scorer), (vec![300, 0, 300], 600));
}

#[test]
fn a_pair_pick_s_second_target_books_its_decode_and_not_the_prefill_again() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    // Each target of a pair pick names the one prefill, the first row,
    // with another decode: the second row, then the third.
    let target = |decode: u32| Target {
        position: 0,
        id: pool[0].id,
        dp_rank: 0,
        pair: Some(Pair {
            decode_position: decode,
            decode_id: pool[decode as usize].id,
            ..Pair::default()
        }),
    };
    let mut targets = Targets::new();
    targets.push(target(1));
    targets.push(target(2));
    let decision = Decision::Pick(Pick {
        targets,
        ..Pick::default()
    });
    let lease = |side: Side, booked: WorkerId, index: u8| Lease {
        side,
        target_index: index,
        booked,
        tokens: 300,
        issued_micros: SECOND,
        ..Lease::default()
    };
    // The prefill side: its issue books the prefill, and a dispatch on
    // the second target names that prefill again.
    let prefill = pool[0].id;
    scorer.on_decision_issued(&lease(Side::Prefill, prefill, 0), &decision);
    scorer.on_dispatched(
        &lease(Side::Prefill, prefill, 1),
        &target(2),
        Side::Prefill,
        250,
    );
    assert_eq!(loads(&scorer), (vec![300, 0, 0], 300));
    // The decode side: its issue books the first target's decode, and
    // a dispatch on the second target books that target's own.
    let decode = pool[1].id;
    scorer.on_decision_issued(&lease(Side::Decode, decode, 0), &decision);
    assert_eq!(loads(&scorer), (vec![300, 300, 0], 600));
    scorer.on_dispatched(
        &lease(Side::Decode, decode, 1),
        &target(2),
        Side::Decode,
        250,
    );
    assert_eq!(loads(&scorer), (vec![300, 300, 300], 900));
}

/// A pair's decode that goes out with no decode call is told issued at
/// its dispatch, on the target the dispatch took, and then dispatched
/// as booked there, as a store tells both words: no position, the
/// issue dated when the store is told. One booking stands on the
/// worker that serves, whichever target was taken, and a booking made
/// since the dispatch stands with it. Dated as the gateway dates the
/// dispatch, more than a window behind that booking, the same issue
/// would end it: a clock that went back.
#[test]
fn a_decode_issued_at_its_dispatch_books_once_and_ends_no_newer_booking() {
    // The pair was decided at one second and its decode sent 250
    // microseconds later. The store is told at three seconds, behind
    // a booking of the fourth row made then, under a window of one.
    let datings = [(3 * SECOND, 1_000_u64), (SECOND + 250, 0)];
    for taken in [0_u8, 1] {
        for (issued_micros, stands) in datings {
            let scorer = scorer(NEVER, 1.5, 1);
            let pool = pool(4);
            led(&scorer, &pool, 0, SECOND);
            book(&scorer, &pool, 3, 1_000, 3 * SECOND);
            // Each target names the one prefill, the first row, with
            // another decode: the second row, then the third.
            let target = |decode: usize| Target {
                id: pool[0].id,
                pair: Some(Pair {
                    decode_id: pool[decode].id,
                    ..Pair::default()
                }),
                ..Target::default()
            };
            let mut targets = Targets::new();
            targets.push(target(1));
            targets.push(target(2));
            let decision = Decision::Pick(Pick {
                targets,
                ..Pick::default()
            });
            let serves = 1 + usize::from(taken);
            let lease = |issued_micros: u64| Lease {
                side: Side::Decode,
                target_index: taken,
                booked: pool[serves].id,
                tokens: 300,
                issued_micros,
                ..Lease::default()
            };
            scorer.on_decision_issued(&lease(issued_micros), &decision);
            let sent = Target {
                id: pool[serves].id,
                ..Target::default()
            };
            scorer.on_dispatched(&lease(SECOND), &sent, Side::Decode, 250);
            let mut booked = vec![0, 0, 0, stands];
            booked[serves] = 300;
            let read = loads(&scorer);
            assert_eq!(
                read,
                (booked, 300 + stands),
                "target {taken} at {issued_micros}"
            );
        }
    }
}

/// A pair's decode side as the store files it. The lease the store
/// derives for the side names the side's own worker as the one its
/// issue booked: a dispatch on another rank of that worker leaves the
/// booking of the issue as it stands.
#[test]
fn a_decode_side_s_dispatch_on_another_rank_of_its_worker_books_nothing() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    // Both targets of the pick name the one prefill, the first row,
    // and two ranks of one decode, the second row.
    let mut targets = Targets::new();
    for rank in [0, 1] {
        targets.push(Target {
            position: 0,
            id: pool[0].id,
            dp_rank: 0,
            pair: Some(Pair {
                decode_position: 1,
                decode_id: pool[1].id,
                decode_dp_rank: rank,
                ..Pair::default()
            }),
        });
    }
    let picked = crate::store::PickRecord::of_decodes(&targets);
    let decision = Decision::Pick(Pick {
        targets,
        ..Pick::default()
    });
    // The record's own lease as its decider issues it: it books the
    // pick's first target, the prefill.
    let prefill = Lease {
        side: Side::Prefill,
        target_index: 0,
        booked: pool[0].id,
        tokens: 300,
        issued_micros: SECOND,
        ..Lease::default()
    };
    let side = crate::store::DecodeSide {
        target: pool[1].id,
        picked,
        session: view::Session::NONE,
        session_verdict: crate::engine::SessionVerdict::None,
        state: crate::store::LeaseState::ISSUED,
        tokens: 300,
        target_index: 0,
    };
    let decode = side.lease(&prefill);
    scorer.on_decision_issued(&decode, &decision);
    assert_eq!(loads(&scorer), (vec![0, 300, 0], 300));
    // The side's first dispatch, on the second target, as the store
    // tells it: the side's lease with the frame's index, on the side's
    // own worker.
    let told = Lease {
        target_index: 1,
        ..decode
    };
    let on = Target {
        id: pool[1].id,
        dp_rank: 1,
        ..Target::default()
    };
    scorer.on_dispatched(&told, &on, Side::Decode, 250);
    assert_eq!(loads(&scorer), (vec![0, 300, 0], 300), "booked once");
}

#[test]
fn a_decode_side_lease_books_its_pair_s_decode_and_not_the_prefill() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(2);
    led(&scorer, &pool, 0, SECOND);
    let paired = |decode_position: u32, decode_id: WorkerId| {
        let mut targets = Targets::new();
        targets.push(Target {
            position: 0,
            id: pool[0].id,
            dp_rank: 0,
            pair: Some(Pair {
                decode_position,
                decode_id,
                ..Pair::default()
            }),
        });
        Decision::Pick(Pick {
            targets,
            ..Pick::default()
        })
    };
    let lease = |side: Side| Lease {
        side,
        tokens: 300,
        issued_micros: SECOND,
        ..Lease::default()
    };
    scorer.on_decision_issued(&lease(Side::Decode), &paired(7, WorkerId(907)));
    assert_eq!(
        loads(&scorer),
        (vec![0, 0], 0),
        "a decode this partition does not list books nothing, and not on the pair's prefill"
    );
    scorer.on_decision_issued(&lease(Side::Prefill), &paired(7, WorkerId(907)));
    assert_eq!(
        loads(&scorer),
        (vec![300, 0], 300),
        "the prefill side books the prefill"
    );
    scorer.on_decision_issued(&lease(Side::Decode), &paired(1, pool[1].id));
    assert_eq!(
        loads(&scorer),
        (vec![300, 300], 600),
        "a decode the partition lists is booked as itself"
    );
}
