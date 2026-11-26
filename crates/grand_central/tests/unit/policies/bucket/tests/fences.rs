//! The fence on a worker that left. A view older than the leave lists
//! it nowhere, until the feed names it again or the fence lapses.

use super::*;

#[test]
fn a_view_older_than_a_leave_lists_nobody_splits_nothing_and_places_the_others() {
    let (scorer, pool) = three_classes();
    led(&scorer, &pool, 0, 15 * SECOND);
    scorer.on_membership(&Membership::Removed(pool[1].id));
    assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
    // A decision whose view was captured before the leave still shows
    // the worker. It joins nowhere, the ranges stand, and each size
    // goes to the worker that owns it, at the row the view shows it in.
    assert_eq!(led(&scorer, &pool, 300, 16 * SECOND), Some(0));
    assert_eq!(led(&scorer, &pool, 2_000, 16 * SECOND), Some(2));
    assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
    assert_eq!(loads(&scorer), (vec![1_000, 3_000], 6_000));
    assert_eq!(scorer.listed.load().len(), 2, "the index names the two");
    // The leaver alone among the survivors: nobody is seated, and the
    // picker's draw places the request.
    assert_eq!(
        led_by(&scorer, &prefill("m", 300, 16 * SECOND), &pool, &[1]),
        None
    );
    assert_eq!(ranges(&scorer).len(), 2);
}

#[test]
fn a_partition_first_seen_through_a_view_older_than_a_leave_lists_the_others() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    // The worker left before the model's first prefill decision, whose
    // view was captured before the leave.
    scorer.on_membership(&Membership::Removed(pool[1].id));
    assert_eq!(led(&scorer, &pool, 3_000, SECOND), Some(2));
    assert_eq!(ranges(&scorer), [(0, 0, 2_047), (1, 2_048, u32::MAX)]);
    assert_eq!(loads(&scorer), (vec![0, 0], 0));
}

/// The fence takes a leave's word before the leaver's partition forgets
/// it, and holds no lock across the walk.
#[test]
fn a_leaver_is_fenced_while_its_partition_still_lists_it() {
    let scorer = Arc::new(scorer(NEVER, 1.5, 5));
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    let gone = pool[1].id;
    let partition = of(&scorer, "m");
    // A decision holds the leaver's partition while the feed's call
    // runs: the call fences the leaver and waits for the partition.
    let held = partition.lock();
    let feed = {
        let scorer = Arc::clone(&scorer);
        std::thread::spawn(move || scorer.on_membership(&Membership::Removed(gone)))
    };
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    let mut fenced = false;
    while !fenced && Instant::now() < deadline {
        fenced = scorer
            .fence
            .try_lock_for(std::time::Duration::from_millis(100))
            .is_some_and(|fence| fence.holds(gone));
        std::thread::yield_now();
    }
    assert!(
        held.by_id.contains_key(&gone),
        "the held partition lists it"
    );
    assert!(fenced, "no fence on the leaver while the walk waited");
    // The call holds no fence while it waits for the partition: the
    // fence stays free for a decision that lists.
    let free = (0..10).all(|_| {
        std::thread::sleep(std::time::Duration::from_millis(20));
        scorer
            .fence
            .try_lock_for(std::time::Duration::from_millis(50))
            .is_some()
    });
    assert!(free, "the fence is held across the walk");
    drop(held);
    feed.join().expect("the feed's call");
    assert!(!partition.lock().by_id.contains_key(&gone));
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer).0.len(), 2, "an older view lists it nowhere");
}

/// A snapshot fences its leavers before its walk reads the table: a
/// partition a decision makes while the walk waits lists none of them.
#[test]
fn a_snapshot_s_leaver_is_fenced_before_a_partition_made_during_its_walk() {
    let scorer = Arc::new(scorer(NEVER, 1.5, 5));
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    let gone = pool[1].id;
    let stay = [pool[0].id, pool[2].id];
    let partition = of(&scorer, "m");
    // A decision holds the leaver's partition while the feed's call
    // runs: the walk has read the table and waits for the partition.
    let held = partition.lock();
    let feed = {
        let scorer = Arc::clone(&scorer);
        std::thread::spawn(move || scorer.on_membership(&Membership::Snapshot(&stay)))
    };
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    let mut fenced = false;
    while !fenced && Instant::now() < deadline {
        fenced = scorer
            .fence
            .try_lock_for(std::time::Duration::from_millis(100))
            .is_some_and(|fence| fence.holds(gone));
        std::thread::yield_now();
    }
    // Another model's first decision, over a view older than the
    // snapshot, makes its partition behind the walk's read of the
    // table: the walk never visits it.
    let _ = led_by(&scorer, &prefill("n", 0, SECOND), &pool, &all(&pool));
    drop(held);
    feed.join().expect("the feed's call");
    let listed = |model: &str| -> Vec<WorkerId> {
        of(&scorer, model)
            .lock()
            .members
            .iter()
            .map(|member| member.id)
            .collect()
    };
    assert_eq!(listed("n"), stay, "the partition made during the walk");
    assert_eq!(listed("m"), stay, "the partition the walk waited for");
    assert!(fenced, "no fence on the leaver while the walk waited");
}

/// A worker a decision was listing when the snapshot's word came is in
/// no index yet: the walk fences it before its partition forgets it.
#[test]
fn a_snapshot_fences_a_leaver_its_partition_lists_and_the_index_does_not() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    led(&scorer, &pool[..2], 0, SECOND);
    // The third row joins under the partition's guard, as at a
    // decision that has not written the index yet.
    of(&scorer, "m").lock().list(&pool, |_| false);
    let gone = pool[2].id;
    assert!(!scorer.listed.load().contains_key(&gone));
    scorer.on_membership(&Membership::Snapshot(&[pool[0].id, pool[1].id]));
    assert!(scorer.fence.lock().holds(gone), "the walk fenced nobody");
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer).0.len(), 2, "an older view lists it nowhere");
}

/// A decision asks the fence of every row of its view the partition does
/// not list, the filters' dropped rows too: a fenced row the filters
/// dropped joins nowhere when another row of the view joins.
#[test]
fn a_fenced_row_the_filters_dropped_joins_nowhere_beside_a_new_row() {
    let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
    let gone = pool[1].id;
    scorer.on_membership(&Membership::Removed(gone));
    let wider = self::pool(3);
    let _ = led_by(&scorer, &prefill("m", 0, 2 * SECOND), &wider, &[0, 2]);
    let listed: Vec<WorkerId> = of(&scorer, "m")
        .lock()
        .members
        .iter()
        .map(|member| member.id)
        .collect();
    assert_eq!(listed, [wider[0].id, wider[2].id]);
}

/// A view older than two leaves places the rows behind the leavers as
/// a view taken after the leaves does, whether the filters kept a
/// leaver's row or dropped it.
#[test]
fn a_view_older_than_two_leaves_places_the_rows_behind_them_as_a_new_view_does() {
    let (scorer, pool) = booked(NEVER, 1.5, &[10, 20, 30, 40, 50, 60]);
    for row in [1, 3] {
        scorer.on_membership(&Membership::Removed(pool[row].id));
    }
    let mut new: Vec<Candidate> = [0, 2, 4, 5].map(|row: usize| pool[row]).to_vec();
    for (position, row) in new.iter_mut().enumerate() {
        row.position = position as u32;
    }
    // The old view's first leaver passed the filters and its second did
    // not.
    let kept = [0, 1, 2, 4, 5];
    for tokens in [0, 1_000, 2_000, 3_000, 4_000] {
        let facts = prefill("m", tokens, 2 * SECOND);
        let by_new = led(&scorer, &new, tokens, 2 * SECOND).map(|row| new[row as usize].id);
        let by_old = led_by(&scorer, &facts, &pool, &kept).map(|row| pool[row as usize].id);
        assert_eq!(by_old, by_new, "{tokens} tokens");
        assert!(by_old.is_some(), "{tokens} tokens have an owner");
    }
    // The leavers' bookings stay in the window and on no worker.
    assert_eq!(loads(&scorer), (vec![10, 30, 50, 60], 210));
}

/// The fenced rows a partition keeps are its last listing's alone: the
/// next listing starts from none, so the list never outgrows a view.
#[test]
fn the_fenced_rows_a_partition_keeps_are_its_last_listings_alone() {
    let (scorer, pool) = booked(NEVER, 1.5, &[10, 20, 30]);
    let gone = pool[1].id;
    scorer.on_membership(&Membership::Removed(gone));
    for _ in 0..3 {
        led(&scorer, &pool, 0, 2 * SECOND);
        assert_eq!(of(&scorer, "m").lock().fenced, [(1, gone)]);
    }
    // The feed names the worker again: the old view's next decision
    // lists it, and that listing keeps no fenced row.
    scorer.on_membership(&Membership::Added(gone));
    led(&scorer, &pool, 0, 2 * SECOND);
    assert!(of(&scorer, "m").lock().fenced.is_empty());
    assert_eq!(loads(&scorer).0.len(), 3);
}

#[test]
fn the_feed_naming_a_worker_again_lifts_its_fence() {
    // Added: the worker joins from the next view that shows it.
    let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
    let gone = pool[1].id;
    scorer.on_membership(&Membership::Removed(gone));
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100], 140), "fenced");
    scorer.on_membership(&Membership::Added(gone));
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 0], 140), "named again");
    // A join told again of a worker that stands changes nothing.
    scorer.on_membership(&Membership::Added(gone));
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 0], 140), "named twice");

    // A replacing handle: the id comes back as the new handle of a
    // worker this model never listed.
    let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
    scorer.on_membership(&Membership::Removed(gone));
    scorer.on_membership(&Membership::Replaced {
        old: WorkerId(7),
        new: gone,
    });
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 0], 140));

    // A snapshot that names it.
    let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
    scorer.on_membership(&Membership::Removed(gone));
    scorer.on_membership(&Membership::Snapshot(&[pool[0].id, gone]));
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 0], 140));
}

#[test]
fn a_replaced_handle_and_a_snapshot_s_leavers_are_fenced_as_a_removed_worker_is() {
    // The old handle of a replacement: a view older than the word still
    // shows it, and it joins as nobody.
    let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
    let fresh = WorkerId(900);
    scorer.on_membership(&Membership::Replaced {
        old: pool[1].id,
        new: fresh,
    });
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 40], 140), "it stands once");
    // A replacement by the same handle fences nothing.
    scorer.on_membership(&Membership::Replaced {
        old: fresh,
        new: fresh,
    });
    assert!(!scorer.fence.lock().is_retired(fresh, Instant::now()));

    // The workers a snapshot does not name.
    let (scorer, pool) = three_classes();
    scorer.on_membership(&Membership::Snapshot(&[pool[2].id]));
    assert_eq!(led(&scorer, &pool, 300, 12 * SECOND), Some(2));
    assert_eq!(loads(&scorer), (vec![3_000], 6_000));
}

#[test]
fn the_fence_lapses_after_its_window_and_the_worker_joins_from_a_view_again() {
    let now = Instant::now();
    // A leave as the membership call makes it, dated half a fence back.
    let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
    let gone = pool[1].id;
    let half = now
        .checked_sub(RETIRED_WORKER_TTL / 2)
        .expect("half a fence back");
    scorer.fence.lock().retire(gone, half);
    of(&scorer, "m").lock().remove(gone);
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100], 140), "still fenced");

    // The same leave a whole fence back: the id may be a new worker's.
    let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
    let whole = now
        .checked_sub(RETIRED_WORKER_TTL)
        .expect("a whole fence back");
    scorer.fence.lock().retire(gone, whole);
    of(&scorer, "m").lock().remove(gone);
    led(&scorer, &pool, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 0], 140), "lapsed");
    assert!(
        scorer.fence.lock().is_empty(),
        "the decision that listed forgot the lapsed retirement"
    );
}
