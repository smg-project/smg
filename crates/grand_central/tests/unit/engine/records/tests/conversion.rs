//! The conversion of a list's reports under a snapshot: the decision that
//! converts, what a decision beside it reads, and the look-ups draws pay.

use super::*;

#[test]
fn a_lists_reports_are_converted_once_a_snapshot_by_the_decision_that_holds_it() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(3);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let begun = Instant::now();
    let at = |millis: u64| begun + Duration::from_millis(millis);
    assert!(held(&pool).is_none() && pool.conversions() == 0);

    // A decision that holds no snapshot converts nothing.
    let live = Live::at(spec.engine(), &clock, at(0), None, None);
    let _ = pool.write(&workers, &live, &mut rows);
    assert!(held(&pool).is_none() && pool.conversions() == 0);

    // The first decision under a snapshot converts the list's reports, and
    // the list holds them under the snapshot's name.
    let (first, _) = reports(&[100, 200, 300], Duration::from_millis(40));
    let live = Live::at(spec.engine(), &clock, at(1), Some(&first), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert!(facts.reported);
    assert_eq!(
        written[1].load.expect("a report").waiting_uncached_tokens,
        200
    );
    let converted = held(&pool).expect("the rows of the first snapshot");
    assert!(converted.is_of(&first));
    assert_eq!(pool.conversions(), 1);

    // The decisions behind it under the same snapshot read those rows: no
    // report is converted again, and the rows are the same value.
    for millis in 2..6 {
        let live = Live::at(spec.engine(), &clock, at(millis), Some(&first), None);
        let (written, _) = pool.write(&workers, &live, &mut rows);
        assert_eq!(
            written[2].load.expect("a report").waiting_uncached_tokens,
            300
        );
    }
    assert_eq!(pool.conversions(), 1);
    assert!(Arc::ptr_eq(
        &converted,
        &held(&pool).expect("the same rows")
    ));

    // The next snapshot replaces the rows whole: one conversion more.
    let (second, _) = reports(&[900, 800], Duration::from_millis(10));
    let live = Live::at(spec.engine(), &clock, at(10), Some(&second), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert_eq!(
        written[0].load.expect("a report").waiting_uncached_tokens,
        900
    );
    assert!(written[2].load.is_none() && !facts.reported);
    let replaced = held(&pool).expect("the rows of the second snapshot");
    assert!(replaced.is_of(&second) && !replaced.is_of(&first));
    assert_eq!(pool.conversions(), 2);

    // The rows name their snapshot and do not hold it: the first snapshot
    // is gone with its last holder.
    let name = Arc::downgrade(&first);
    drop(first);
    assert!(name.upgrade().is_none());
    drop(second);
    assert_eq!(replaced.snapshot.strong_count(), 0);
}

/// A document that declares the two token counts a report offers by name,
/// and reads no load report. Where `read`, a stage of its one profile reads
/// both counts. Where not, no stage reads either.
fn by_signals(read: bool) -> AppliedSpec {
    let room = r#",{"has-room":{"used":{"signal.kv_tokens_used":1},
                    "capacity":"signal.kv_tokens_capacity"}}"#;
    let document = format!(
        r#"{{"apiVersion":"smg.policy/v1",
            "signals":{{"kv_tokens_capacity":{{"unit":"tokens"}},
                       "kv_tokens_used":{{"unit":"tokens"}}}},
            "profiles":{{"p":{{"filter":["healthy"{}],"pick":"random"}}}}}}"#,
        if read { room } else { "" },
    );
    applied_of(serde_json::from_str(&document).expect("a well-formed document"))
}

/// A snapshot in which each worker of `used`, by position, uses that many
/// of a thousand tokens, sampled now.
fn counts(used: &[i32]) -> (Arc<LoadSnapshot>, Instant) {
    let sampled_at = Instant::now();
    let loads = used
        .iter()
        .enumerate()
        .map(|(at, &used)| {
            let mut report = one_rank_load(0, 0.1, 1_000.0);
            report.sampled_at = Some(sampled_at);
            for rank in &mut report.loads {
                rank.num_used_tokens = used;
                rank.max_total_num_tokens = 1_000;
            }
            (url(at), report)
        })
        .collect();
    (LoadSnapshot::from_loads_for_test(loads), sampled_at)
}

/// Each row's capacity and what is used of it, as `cells` hold them for
/// three rows under [`by_signals`]: the capacity's column stands first.
fn counted(cells: &SignalCells) -> Vec<Option<(f64, f64)>> {
    (0..3)
        .map(|row| {
            let row = cells.row(row).expect("a row of cells");
            (!row[0].is_none()).then(|| (row[0].value, row[1].value))
        })
        .collect()
}

/// A conversion of a list's reports makes the list's signal cells under a
/// document a stage of which reads a named signal: a cell a worker a
/// signal, each value at its worker's place under its column, dated on the
/// decision's clock. A decision is lent those cells under their snapshot.
/// One that holds another snapshot and is not the one to convert is lent
/// its own snapshot's, in the thread's buffer, which takes them back. One
/// that holds no snapshot is lent none. Under a document that declares the
/// signals and reads neither, a conversion makes no cell.
#[test]
fn a_conversion_makes_the_lists_signal_cells_under_a_document_that_reads_one() {
    let spec = by_signals(true);
    let clock = DecisionClock::new();
    let workers = fleet(3);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let begun = Instant::now();
    let at = |millis: u64| begun + Duration::from_millis(millis);

    // A decision that writes every record converts: the third worker has
    // no report, and its cells hold no value.
    let (first, sampled_at) = counts(&[100, 200]);
    let live = Live::at(spec.engine(), &clock, at(5), Some(&first), None);
    let _ = pool.write(&workers, &live, &mut Vec::new());
    let converted = held(&pool).expect("the rows of the first snapshot");
    assert!(converted.cells.fits(3) && converted.cells.signals() == 2);
    let stated = [Some((1_000.0, 100.0)), Some((1_000.0, 200.0)), None];
    assert_eq!(counted(&converted.cells), stated);
    let dated = converted.cells.cells().iter().take(4);
    assert!(dated
        .map(|cell| cell.sampled_at_micros)
        .eq([clock.micros(sampled_at); 4]));
    let lent = pool.cells(&workers, &live).expect("held cells");
    let own = ptr::eq(lent.cells().expect("cells"), &converted.cells);
    assert!(own, "the cells the conversion left");

    // A decision that began earlier and holds another snapshot leaves the
    // list's rows in place, and reads its own snapshot's values. They
    // stand in the thread's buffer, which has its room back afterwards.
    let (second, _) = counts(&[700, 800, 900]);
    let earlier = Live::at(spec.engine(), &clock, at(1), Some(&second), None);
    let lent = pool.cells(&workers, &earlier).expect("held cells");
    let all = [700.0, 800.0, 900.0].map(|used| Some((1_000.0, used)));
    assert_eq!(counted(lent.cells().expect("cells")), all);
    assert!(held(&pool).is_some_and(|rows| rows.is_of(&first)));
    assert_eq!(CELLS.with(Cell::take).cells_mut().capacity(), 0);
    drop(lent);
    assert!(CELLS.with(Cell::take).cells_mut().capacity() >= 6);
    // A later one that holds it converts, and is lent what it converted.
    let later = Live::at(spec.engine(), &clock, at(9), Some(&second), None);
    let lent = pool.cells(&workers, &later).expect("held cells");
    assert_eq!(counted(lent.cells().expect("cells")), all);
    assert!(held(&pool).is_some_and(|rows| rows.is_of(&second)));
    assert_eq!(pool.conversions(), 2);
    // Under no snapshot a decision is lent no cells.
    let bare = Live::at(spec.engine(), &clock, at(10), None, None);
    assert!(pool.cells(&workers, &bare).is_none());

    // A document that declares the two signals and reads neither: the
    // conversion makes no cell.
    let unread = by_signals(false);
    let pool = PoolRecords::new(&workers, &unread, &clock);
    let live = Live::at(unread.engine(), &clock, at(5), Some(&first), None);
    let _ = pool.write(&workers, &live, &mut Vec::new());
    let converted = held(&pool).expect("the rows of the first snapshot");
    assert!(converted.cells.cells().is_empty(), "no cell is made");
}

#[test]
fn a_decision_that_holds_another_snapshot_reads_its_own_reports() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let begun = Instant::now();
    let at = |millis: u64| begun + Duration::from_millis(millis);
    let (old, _) = reports(&[100, 200], Duration::from_millis(40));
    let (new, _) = reports(&[900], Duration::from_millis(10));

    // The list holds the new snapshot's rows, converted by a decision that
    // began at the tenth millisecond.
    let live = Live::at(spec.engine(), &clock, at(10), Some(&new), None);
    let _ = pool.write(&workers, &live, &mut rows);
    let converted = held(&pool).expect("the rows of the new snapshot");
    assert!(converted.is_of(&new));

    // A decision that began before it still holds the old snapshot. It
    // reads the old snapshot's reports, every one, never the list's rows,
    // and it leaves the rows in place: its snapshot is no newer.
    let live = Live::at(spec.engine(), &clock, at(9), Some(&old), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert_eq!(
        written[0].load.expect("a report").waiting_uncached_tokens,
        100
    );
    assert_eq!(
        written[1].load.expect("a report").waiting_uncached_tokens,
        200
    );
    assert!(facts.reported, "the old snapshot holds a report of each");
    assert!(Arc::ptr_eq(
        &converted,
        &held(&pool).expect("the same rows")
    ));
    assert_eq!(pool.conversions(), 1);
    // One record read alone under the old snapshot is the old report too.
    let alone = pool.read(&workers, &live, 1).expect("a worker");
    assert_eq!(text(&alone), text(&written[1]));

    // The decision under the new snapshot still reads the new rows.
    let live = Live::at(spec.engine(), &clock, at(11), Some(&new), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    assert_eq!(
        written[0].load.expect("a report").waiting_uncached_tokens,
        900
    );
    assert!(written[1].load.is_none() && !facts.reported);
    assert_eq!(pool.conversions(), 1);
}

#[test]
fn a_decision_that_waited_ahead_of_the_claim_leaves_the_rows_of_a_later_decision() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let begun = Instant::now();
    let at = |millis: u64| begun + Duration::from_millis(millis);
    let (first, _) = reports(&[100, 200], Duration::from_millis(40));
    let (older, _) = reports(&[300, 400], Duration::from_millis(30));
    let (newer, _) = reports(&[900, 950], Duration::from_millis(10));

    // The list holds the first snapshot's rows.
    let live = Live::at(spec.engine(), &clock, at(1), Some(&first), None);
    let _ = pool.write(&workers, &live, &mut rows);
    assert_eq!(pool.conversions(), 1);

    // A decision under the older of the two snapshots that follow began
    // after those rows were converted, so it may convert. It has read the
    // slot and has not taken the list's claim yet.
    let waiting = Live::at(spec.engine(), &clock, at(2), Some(&older), None);
    let read_first = pool.reported.load();

    // Meanwhile a decision that began later converts the newer snapshot.
    let live = Live::at(spec.engine(), &clock, at(3), Some(&newer), None);
    let _ = pool.write(&workers, &live, &mut rows);
    let converted = held(&pool).expect("the rows of the newer snapshot");
    assert!(converted.is_of(&newer));
    assert_eq!(pool.conversions(), 2);

    // The decision that waited goes on with what it read and pays the
    // list's length, as one that writes every record does. Under the claim it
    // finds the rows of a decision that began later: it converts nothing and
    // leaves them in place.
    let again = pool.convert(&workers, &older, &waiting, &read_first, 2);
    assert_eq!(pool.conversions(), 2, "its snapshot is no newer");
    assert!(again.is_none());
    assert!(Arc::ptr_eq(
        &converted,
        &held(&pool).expect("the same rows")
    ));

    // It writes its records from its own snapshot all the same.
    let (written, _) = pool.write(&workers, &waiting, &mut rows);
    assert_eq!(
        written[0].load.expect("a report").waiting_uncached_tokens,
        300
    );
    assert_eq!(
        written[1].load.expect("a report").waiting_uncached_tokens,
        400
    );
    assert_eq!(pool.conversions(), 2);

    // So the next decision under the newer snapshot converts nothing.
    let live = Live::at(spec.engine(), &clock, at(4), Some(&newer), None);
    let (written, _) = pool.write(&workers, &live, &mut rows);
    assert_eq!(
        written[0].load.expect("a report").waiting_uncached_tokens,
        900
    );
    assert_eq!(pool.conversions(), 2);

    // The rule is of the two instants alone. With the same read, a decision
    // that began at the instant of the rows' conversion converts nothing,
    // and one that began later converts its own snapshot.
    let equal = Live::at(spec.engine(), &clock, at(3), Some(&older), None);
    assert!(pool
        .convert(&workers, &older, &equal, &read_first, 2)
        .is_none());
    assert_eq!(pool.conversions(), 2);
    let later = Live::at(spec.engine(), &clock, at(5), Some(&older), None);
    let replaced = pool
        .convert(&workers, &older, &later, &read_first, 2)
        .expect("a decision that began later converts");
    assert!(replaced.is_of(&older));
    assert_eq!(pool.conversions(), 3);
}

#[test]
fn a_decision_that_waited_ahead_of_the_claim_leaves_the_lists_count_of_look_ups() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let begun = Instant::now();
    let at = |millis: u64| begun + Duration::from_millis(millis);
    let (first, _) = reports(&[100, 200], Duration::from_millis(40));
    let (older, _) = reports(&[300, 400], Duration::from_millis(30));
    let (newer, _) = reports(&[900, 950], Duration::from_millis(10));
    let (fresh, _) = reports(&[500, 600], Duration::from_millis(5));

    // The list holds the first snapshot's rows. A decision under the older
    // snapshot has read the slot and waits ahead of the list's claim, and a
    // decision that began later converts the newer snapshot.
    let live = Live::at(spec.engine(), &clock, at(1), Some(&first), None);
    let _ = pool.write(&workers, &live, &mut rows);
    let waiting = Live::at(spec.engine(), &clock, at(2), Some(&older), None);
    let read_first = pool.reported.load();
    let live = Live::at(spec.engine(), &clock, at(3), Some(&newer), None);
    let _ = pool.write(&workers, &live, &mut rows);
    assert_eq!(pool.conversions(), 2);

    // A draw under a fourth snapshot pays one of the list's two look-ups.
    let drawn = Live::at(spec.engine(), &clock, at(4), Some(&fresh), None);
    assert!(pool
        .convert(&workers, &fresh, &drawn, &pool.reported.load(), 1)
        .is_none());

    // The decision that waited pays one under its older snapshot. Under the
    // claim it finds the rows of a decision that began later, so the list's
    // count of look-ups stays that of the fourth snapshot.
    assert!(pool
        .convert(&workers, &older, &waiting, &read_first, 1)
        .is_none());
    assert_eq!(pool.conversions(), 2);

    // So a second draw under the fourth snapshot pays the last of the
    // list's length and converts.
    let second = Live::at(spec.engine(), &clock, at(5), Some(&fresh), None);
    let converted = pool
        .convert(&workers, &fresh, &second, &pool.reported.load(), 1)
        .expect("the second look-up of the fourth snapshot converts");
    assert!(converted.is_of(&fresh));
    assert_eq!(pool.conversions(), 3);
}

#[test]
fn a_decision_that_meets_a_conversion_under_way_converts_nothing_and_waits_for_nothing() {
    let spec = applied("least_load");
    let clock = DecisionClock::new();
    let workers = fleet(2);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let mut rows = Vec::new();
    let (loads, _) = reports(&[100, 200], Duration::from_millis(40));

    // Another decision holds the list's claim: it is converting.
    let claim = pool.converting.try_lock().expect("the claim is free");
    assert!(
        pool.converting.try_lock().is_none(),
        "one decision holds it"
    );
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    let (written, facts) = pool.write(&workers, &live, &mut rows);
    // This one wrote its own records from its own snapshot, and left
    // nothing behind.
    assert_eq!(
        written[1].load.expect("a report").waiting_uncached_tokens,
        200
    );
    assert!(facts.reported);
    assert!(held(&pool).is_none() && pool.conversions() == 0);

    // The claim is given back at its drop, and the next decision converts.
    drop(claim);
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    let _ = pool.write(&workers, &live, &mut rows);
    assert!(held(&pool).is_some_and(|converted| converted.is_of(&loads)));
    assert_eq!(pool.conversions(), 1);
    assert!(pool.converting.try_lock().is_some(), "given back");
}

/// One draw of `spec`'s engine over `pool` under `loads`, by `seed`.
fn drawn(
    spec: &AppliedSpec,
    clock: &DecisionClock,
    pool: &PoolRecords,
    workers: &[Arc<dyn Worker>],
    loads: &Arc<LoadSnapshot>,
    seed: u64,
) -> Decision {
    let live = Live::at(spec.engine(), clock, Instant::now(), Some(loads), None);
    let mut facts = RequestFacts { seed, ..single() };
    let cx = Constraints::default();
    decide(spec.engine(), pool, workers, &live, &mut facts, &cx)
}

/// Draws convert a list's reports once they looked up as many as it holds: a
/// pick of two looks up two a decision, so the 32nd pays the last of the 64 and
/// converts. Later draws copy the list's rows, and a new snapshot counts again.
#[test]
fn draws_convert_a_lists_reports_once_they_looked_up_a_lists_length() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(64);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let queued: Vec<i32> = (0..64).map(|at| at * 10).collect();
    for snapshot in 1..=2 {
        let (loads, _) = reports(&queued, Duration::from_millis(5));
        for seed in 1..=31 {
            let decision = drawn(&spec, &clock, &pool, &workers, &loads, seed);
            assert_eq!(decision.pick().expect("a pick").targets.len(), 2);
            assert_eq!(pool.conversions(), snapshot - 1, "decision {seed}");
        }
        let _ = drawn(&spec, &clock, &pool, &workers, &loads, 32);
        assert_eq!(pool.conversions(), snapshot, "the draw that pays the last");
        assert!(held(&pool).is_some_and(|rows| rows.is_of(&loads)));
        for seed in 33..=200 {
            let _ = drawn(&spec, &clock, &pool, &workers, &loads, seed);
        }
        assert_eq!(pool.conversions(), snapshot, "the rows are copied");
    }
}

/// A snapshot published after one that is gone starts the list's count again,
/// wherever it stands in memory: the list names the snapshot it counted under,
/// so 62 look-ups under one and four under each of 64 later convert nothing.
#[test]
fn a_later_snapshot_does_not_inherit_a_lists_count() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(64);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let queued: Vec<i32> = (0..64).map(|at| at * 10).collect();
    let (first, _) = reports(&queued, Duration::from_millis(5));
    for seed in 1..=31 {
        let _ = drawn(&spec, &clock, &pool, &workers, &first, seed);
    }
    // One more decision would convert. The snapshot goes, its place stays named.
    let place = Arc::as_ptr(&first).addr();
    drop(first);
    for later in 0..64 {
        let (loads, _) = reports(&queued, Duration::from_millis(5));
        if later == 0 {
            assert_ne!(Arc::as_ptr(&loads).addr(), place, "the place is named");
        }
        for seed in [100 + later, 200 + later] {
            let _ = drawn(&spec, &clock, &pool, &workers, &loads, seed);
            assert_eq!(pool.conversions(), 0, "snapshot {later}");
        }
    }
}

/// A draw that finds the list's claim taken counts nothing and waits for
/// nothing: the count starts with the first draw that takes the claim.
#[test]
fn a_draw_that_finds_the_claim_taken_counts_nothing() {
    let (spec, clock) = (applied("power_of_two"), DecisionClock::new());
    let workers = fleet(64);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let queued: Vec<i32> = (0..64).map(|at| at * 10).collect();
    let (loads, _) = reports(&queued, Duration::from_millis(5));
    let claim = pool.converting.try_lock().expect("the claim is free");
    for seed in 1..=40 {
        let _ = drawn(&spec, &clock, &pool, &workers, &loads, seed);
    }
    assert_eq!(claim.records, 0, "80 look-ups, none counted");
    drop(claim);
    for seed in 1..=31 {
        let _ = drawn(&spec, &clock, &pool, &workers, &loads, seed);
    }
    assert_eq!(pool.conversions(), 0, "62 counted");
    let _ = drawn(&spec, &clock, &pool, &workers, &loads, 32);
    assert_eq!(pool.conversions(), 1);
}
