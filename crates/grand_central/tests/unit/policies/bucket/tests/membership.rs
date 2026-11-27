//! Who a partition lists. Its cases cover a new row, a removed worker,
//! a registration's word, a replaced handle and a model's own partition.

use super::*;

/// Each member with its load, each range, and the window's total.
type Listing = (Vec<(WorkerId, u64)>, Vec<(u32, u32, u32)>, u64);

/// The listing of `model`'s partition.
fn listing_of(scorer: &Bucket, model: &str) -> Listing {
    let partition = of(scorer, model);
    let held = partition.lock();
    let members = held
        .members
        .iter()
        .map(|member| (member.id, member.load))
        .collect();
    let ranges = held
        .ranges
        .iter()
        .map(|range| (range.member, range.lower, range.upper))
        .collect();
    (members, ranges, held.total)
}

#[test]
fn a_row_the_partition_never_listed_splits_the_span_again_and_keeps_the_loads() {
    let (scorer, two) = booked(NEVER, 1.5, &[100, 40]);
    led(&scorer, &two, 0, 6 * SECOND);
    assert_eq!(ranges(&scorer), [(0, 0, 40), (1, 41, u32::MAX)], "learned");
    // A third worker in the view: the span in thirds, the loads kept,
    // and the next cut made whatever the share.
    let three = pool(3);
    assert_eq!(led(&scorer, &three, 3_000, 7 * SECOND), Some(2));
    assert_eq!(
        ranges(&scorer),
        [(0, 0, 1_364), (1, 1_365, 2_729), (2, 2_730, u32::MAX)]
    );
    assert_eq!(loads(&scorer), (vec![100, 40, 0], 140));
    assert_eq!(of(&scorer, "m").lock().cut_for, 0);
    // A view that shows fewer rows forgets nobody: the lifecycle prunes.
    assert_eq!(
        led(&scorer, &two, 3_000, 8 * SECOND),
        Some(1),
        "the nearest below"
    );
    assert_eq!(ranges(&scorer).len(), 3);
    // A new row and a cut due at one decision: the listing first, so the
    // cut is over all three.
    let (scorer, _) = booked(NEVER, 1.5, &[100, 40]);
    assert_eq!(led(&scorer, &three, 3_000, 6 * SECOND), Some(2));
    assert_eq!(
        ranges(&scorer),
        [(0, 0, 40), (1, 41, 100), (2, 101, u32::MAX)]
    );
}

#[test]
fn a_removed_worker_s_range_goes_to_the_range_below_and_the_first_s_to_the_one_above() {
    let (scorer, pool) = three_classes();
    led(&scorer, &pool, 0, 15 * SECOND);
    scorer.on_membership(&Membership::Removed(pool[1].id));
    assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
    // Its bookings stay in the window and count toward nobody's load.
    assert_eq!(loads(&scorer), (vec![1_000, 3_000], 6_000));
    assert_eq!(of(&scorer, "m").lock().cut_for, 0, "the next cut is made");
    // A view of the two that stay is placed by the merged ranges.
    let left = [
        pool[0],
        Candidate {
            position: 1,
            ..pool[2]
        },
    ];
    assert_eq!(
        led_by(&scorer, &prefill("m", 300, 16 * SECOND), &left, &[0, 1]),
        Some(0)
    );
    // The bookings of the removed worker leave with their interval.
    book(&scorer, &left, 1, 5, 17 * SECOND);
    assert_eq!(loads(&scorer), (vec![0, 5], 5));

    let (scorer, pool) = three_classes();
    led(&scorer, &pool, 0, 15 * SECOND);
    scorer.on_membership(&Membership::Removed(pool[0].id));
    assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
    assert_eq!(loads(&scorer), (vec![2_000, 3_000], 6_000));
    scorer.on_membership(&Membership::Removed(pool[1].id));
    assert_eq!(
        ranges(&scorer),
        [(0, 0, u32::MAX)],
        "the last owns every size"
    );
    // An id nobody lists changes nothing.
    scorer.on_membership(&Membership::Removed(WorkerId(7)));
    assert_eq!(loads(&scorer), (vec![3_000], 6_000));
}

/// A worker a registration took off a model is unlisted under that
/// model alone and keeps its place, its range and its load under the
/// models the word does not name, where its next booking alone lands.
/// The words of a row's facts unlist nobody, and a model whose last
/// worker left this way takes its partition with it. The word fences
/// nobody: a model that never listed the two lists them from its first
/// view.
#[test]
fn a_worker_a_registration_took_off_a_model_is_unlisted_under_that_model_alone() {
    let scorer = scorer(NEVER, 1.5, 5);
    let (pool_a, pool_b) = (pool(2), pool(4));
    for (model, pool) in [("a", &pool_a), ("b", &pool_b)] {
        led_by(&scorer, &prefill(model, 3_000, SECOND), pool, &all(pool));
    }
    let moved = pool_a[1].id;
    book(&scorer, &pool_a, 1, 30, SECOND);
    let listing = |model: &str| listing_of(&scorer, model);
    let (a, b) = (listing("a"), listing("b"));
    assert_eq!(a.0, [(pool_a[0].id, 0), (moved, 30)]);
    // A row's facts changed: nobody is unlisted.
    for word in [
        Membership::Added(moved),
        Membership::Relabelled(moved),
        Membership::Resized(moved),
    ] {
        scorer.on_membership(&word);
    }
    assert_eq!((listing("a"), listing("b")), (a, b.clone()));
    // Off model a, and off a model no decision listed.
    scorer.on_membership(&Membership::Left {
        id: moved,
        models: &["a".into(), "c".into()],
    });
    assert_eq!(listing("a").0, [(pool_a[0].id, 0)]);
    assert_eq!(listing("b"), b, "a model the word does not name moved");
    // Its next booking lands under the model it serves, and there alone.
    book(&scorer, &pool_b, 1, 5, SECOND);
    assert_eq!(listing("a").2, 30, "booked under the model it left");
    assert_eq!(listing("b").0[1], (moved, 35));
    // The model's last worker leaves the same way, and the partition
    // with it.
    let held = of(&scorer, "a");
    scorer.on_membership(&Membership::Left {
        id: pool_a[0].id,
        models: &["a".into()],
    });
    assert!(scorer.partitions.load().get("a").is_none());
    assert!(held.lock().left, "a decision that holds it takes the next");
    assert_eq!(listing("b").0.len(), 4);
    // Both workers left a model a moment ago, and another lists them.
    led_by(
        &scorer,
        &prefill("d", 3_000, SECOND),
        &pool_a,
        &all(&pool_a),
    );
    assert_eq!(listing("d").0.len(), 2);
}

/// A new incarnation that comes back with a model less is unlisted
/// under that model, by the word said for it behind its replacement's,
/// and takes its predecessor's place, range and load under the rest. A
/// decision of that model over an older view, between the two words
/// and behind them, lists the old handle nowhere.
#[test]
fn a_new_incarnation_off_a_model_takes_its_predecessor_s_place_under_the_rest() {
    let scorer = scorer(NEVER, 1.5, 5);
    let (pool_a, pool_b) = (pool(2), pool(4));
    for (model, pool) in [("a", &pool_a), ("b", &pool_b)] {
        led_by(&scorer, &prefill(model, 3_000, SECOND), pool, &all(pool));
    }
    let (old, new) = (pool_a[1].id, WorkerId(900));
    book(&scorer, &pool_a, 1, 30, SECOND);
    let b = listing_of(&scorer, "b");
    assert_eq!(b.0[1], (old, 30));
    // The replacement's word, then the word for the new handle. The
    // older view shows the old handle, which is fenced by the first.
    let older = |at| led_by(&scorer, &prefill("a", 3_000, at), &pool_a, &all(&pool_a));
    scorer.on_membership(&Membership::Replaced { old, new });
    older(2 * SECOND);
    let between = listing_of(&scorer, "a").0;
    assert!(between.iter().all(|&(id, _)| id != old), "{between:?}");
    scorer.on_membership(&Membership::Left {
        id: new,
        models: &["a".into()],
    });
    older(3 * SECOND);
    assert_eq!(listing_of(&scorer, "a").0, [(pool_a[0].id, 0)]);
    let after = listing_of(&scorer, "b");
    assert_eq!(after.0[1], (new, 30), "its predecessor's place and load");
    assert_eq!((after.0.len(), after.1, after.2), (4, b.1, b.2));
}

/// A new incarnation under another role left every model of its
/// predecessor's list. The word names them all, so the replacement has
/// no place to carry: the new handle is listed nowhere until a view of
/// its own pool names it, and no place passes from one role to another.
#[test]
fn a_new_incarnation_under_another_role_takes_no_place_of_its_predecessor() {
    let scorer = scorer(NEVER, 1.5, 5);
    let (pool_a, pool_b) = (pool(2), pool(4));
    for (model, pool) in [("a", &pool_a), ("b", &pool_b)] {
        led_by(&scorer, &prefill(model, 3_000, SECOND), pool, &all(pool));
    }
    let (old, new) = (pool_a[1].id, WorkerId(900));
    book(&scorer, &pool_a, 1, 30, SECOND);
    // The replacement's word, then the word for the new handle with
    // both models.
    scorer.on_membership(&Membership::Replaced { old, new });
    scorer.on_membership(&Membership::Left {
        id: new,
        models: &["a".into(), "b".into()],
    });
    let (a, b) = (listing_of(&scorer, "a").0, listing_of(&scorer, "b").0);
    assert_eq!((a.len(), b.len()), (1, 3));
    for (id, _) in a.iter().chain(&b) {
        assert!(*id != old && *id != new, "neither handle is listed");
    }
}

#[test]
fn a_row_off_its_member_s_place_is_found_by_its_worker_s_id() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    // The same three workers in another order of the view.
    let turned: Vec<Candidate> = [2, 0, 1]
        .iter()
        .enumerate()
        .map(|(position, &row)| Candidate {
            position: position as u32,
            ..pool[row]
        })
        .collect();
    assert_eq!(led(&scorer, &turned, 3_000, SECOND), Some(0));
    assert_eq!(led(&scorer, &turned, 100, SECOND), Some(1));
    book(&scorer, &turned, 0, 50, SECOND);
    assert_eq!(loads(&scorer), (vec![0, 0, 50], 50));
    assert_eq!(ranges(&scorer).len(), 3, "no newcomer, no split");
    // Two rows of one worker are one member.
    let scorer = self::scorer(NEVER, 1.5, 5);
    let twice = [
        pool[0],
        Candidate {
            position: 1,
            dp_rank: 1,
            ..pool[0]
        },
        Candidate {
            position: 2,
            ..pool[1]
        },
    ];
    led(&scorer, &twice, 0, SECOND);
    assert_eq!(loads(&scorer), (vec![0, 0], 0));
}

#[test]
fn a_replaced_handle_keeps_its_place_and_a_snapshot_keeps_the_workers_it_names() {
    let (scorer, pool) = three_classes();
    led(&scorer, &pool, 0, 15 * SECOND);
    let cut = [(0, 0, 100), (1, 101, 400), (2, 401, u32::MAX)];
    let fresh = WorkerId(900);
    scorer.on_membership(&Membership::Replaced {
        old: pool[1].id,
        new: fresh,
    });
    assert_eq!(ranges(&scorer), cut);
    assert_eq!(loads(&scorer), (vec![1_000, 2_000, 3_000], 6_000));
    // The view's row under the new handle is the member it replaced.
    let mut view = pool.clone();
    view[1].id = fresh;
    assert_eq!(led(&scorer, &view, 300, 16 * SECOND), Some(1));
    assert_eq!(ranges(&scorer), cut, "no newcomer, no split");
    // A booking finds the member under its new handle, and none under
    // the handle it replaced.
    book(&scorer, &view, 1, 5, 16 * SECOND);
    book(&scorer, &pool, 1, 7, 16 * SECOND);
    assert_eq!(loads(&scorer), (vec![1_000, 2_005, 3_000], 6_005));
    // An added, a relabelled or a resized worker changes nothing here.
    for change in [
        Membership::Added(WorkerId(901)),
        Membership::Relabelled(pool[0].id),
        Membership::Resized(pool[0].id),
    ] {
        scorer.on_membership(&change);
    }
    assert_eq!(ranges(&scorer), cut);
    // A snapshot names every worker of every pool: the listed ones it
    // leaves out are forgotten, the others it names are no members.
    scorer.on_membership(&Membership::Snapshot(&[
        pool[0].id,
        pool[2].id,
        WorkerId(77),
    ]));
    assert_eq!(ranges(&scorer), [(0, 0, 400), (1, 401, u32::MAX)]);
    assert_eq!(loads(&scorer), (vec![1_000, 3_000], 6_005));
    assert_eq!(scorer.listed.load().len(), 2, "the index names the two");
}

#[test]
fn each_model_has_a_partition_of_its_own() {
    let scorer = scorer(NEVER, 1.5, 5);
    let (pool_a, pool_b) = (pool(2), pool(4));
    let over = |model, tokens, pool: &[Candidate]| {
        led_by(&scorer, &prefill(model, tokens, SECOND), pool, &all(pool))
    };
    assert_eq!(over("a", 3_000, &pool_a), Some(1));
    assert_eq!(over("b", 3_000, &pool_b), Some(2));
    assert_eq!(
        over("a", 3_000, &pool_a),
        Some(1),
        "the other model listed nothing here"
    );
    // A worker two models list is booked in each of their partitions,
    // and a worker one model lists in that one alone.
    book(&scorer, &pool_a, 1, 30, SECOND);
    book(&scorer, &pool_b, 3, 9, SECOND);
    let booked = |model: &str| {
        let partition = of(&scorer, model);
        let held = partition.lock();
        let loads: Vec<u64> = held.members.iter().map(|member| member.load).collect();
        (loads, held.total)
    };
    assert_eq!(booked("a"), (vec![0, 30], 30));
    assert_eq!(booked("b"), (vec![0, 30, 0, 9], 39));
    let partitions = scorer.partitions.load();
    let mut listed: Vec<(&str, usize)> = partitions
        .iter()
        .map(|(model, partition)| (&**model, partition.lock().members.len()))
        .collect();
    listed.sort_unstable();
    assert_eq!(listed, [("a", 2), ("b", 4)]);
}

#[test]
fn a_handle_a_view_listed_before_the_word_of_its_replacement_stands_once() {
    let (scorer, pool) = booked(NEVER, 1.5, &[100, 40]);
    // A view shows the second worker under its new handle before the
    // word of the replacement: the row joins as a third member.
    let fresh = WorkerId(900);
    let mut view = pool.clone();
    view[1].id = fresh;
    led(&scorer, &view, 0, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 40, 0], 140));
    // The word comes: the worker stands once, at the place the view gave
    // it, with the tokens booked under the handle it replaced, whose
    // member leaves as a removed worker's does.
    scorer.on_membership(&Membership::Replaced {
        old: pool[1].id,
        new: fresh,
    });
    assert_eq!(loads(&scorer), (vec![100, 40], 140));
    assert_eq!(ranges(&scorer), [(0, 0, 2_729), (1, 2_730, u32::MAX)]);
    assert_eq!(scorer.listed.load().len(), 2, "the index names the two");
    // One lease of the worker is one booking.
    book(&scorer, &view, 1, 70, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 110], 210));
    // A replacement by the same handle changes nothing.
    scorer.on_membership(&Membership::Replaced {
        old: fresh,
        new: fresh,
    });
    assert_eq!(loads(&scorer), (vec![100, 110], 210));
    assert_eq!(ranges(&scorer), [(0, 0, 2_729), (1, 2_730, u32::MAX)]);
    // The bookings that moved age off the handle that stays: one booked
    // past the window ends them all.
    book(&scorer, &view, 0, 1, 20 * SECOND);
    assert_eq!(loads(&scorer), (vec![1, 0], 1));
}
