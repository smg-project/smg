//! The rows that stand in a thread's buffer between whole writes: for which
//! list and snapshot they stand, and what a write leaves behind in them.

use super::*;

/// The rows a whole write into a fresh buffer gives: what every write of
/// the same list at the same instant gives, standing rows or none.
fn fresh(pool: &PoolRecords, workers: &[Arc<dyn Worker>], live: &Live<'_>) -> Vec<String> {
    let mut rows = Vec::new();
    let (rows, _) = pool.write(workers, live, &mut rows);
    rows.iter().map(text).collect()
}

#[test]
fn what_was_written_behind_a_write_never_shows_in_the_next() {
    let clock = DecisionClock::new();
    let workers = fleet(3);
    workers[1].increment_routing_key_load("key-a");
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    for binding in &mut document.bindings {
        if let Some(rule) = &mut binding.sessions {
            rule.assignment = Assignment::MinGroup;
        }
    }
    let (grouped, plain) = (applied_of(document), applied("least_load"));
    assert!(grouped.engine().needs().active_sessions);
    let pool = PoolRecords::new(&workers, &grouped, &clock);
    let (loads, _) = reports(&[10, 20, 30], Duration::from_millis(5));
    let at = Instant::now();
    let counting = Live::at(grouped.engine(), &clock, at, Some(&loads), None);
    let live = Live::at(plain.engine(), &clock, at, Some(&loads), None);

    // A request's key counted on each worker, and another key's.
    let keyed = [Some("key-a"), Some("key-b")].map(|counted_key| Live {
        counted_key,
        ..live
    });

    // The thread's own rows. The vetoes a decision writes into them and
    // its count of requests are written over by the next decision, with
    // the rows standing, and so is the count of its key's requests; a
    // count of sessions the next decision does not read is gone from them.
    let (mut rows, mut stands) = (Vec::new(), Standing::default());
    for live in [
        &live, &live, &counting, &counting, &keyed[0], &keyed[1], &live,
    ] {
        let (written, _) = pool.write_over::<true, false>(&workers, live, &mut rows, &mut stands);
        let texts: Vec<_> = written.iter().map(text).collect();
        assert_eq!(texts, fresh(&pool, &workers, live));
        for row in written {
            row.health.stalled = true;
            row.health.breaker_open = true;
            row.in_flight = 99;
        }
    }
}

#[test]
fn a_callers_own_write_leaves_no_rows_standing() {
    let clock = DecisionClock::new();
    let spec = applied("least_load");
    let workers = fleet(3);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let (loads, _) = reports(&[10, 20, 30], Duration::from_millis(5));
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    let fresh = fresh(&pool, &workers, &live);
    // Each write is marked in a field no write changes: where the mark is
    // read again the rows stood. Marked or not, they read as a fresh write.
    let stood = |rows: &mut [Candidate]| {
        let stood = mem::replace(&mut rows[0].zone, 0) == 9;
        assert_eq!(rows.iter().map(text).collect::<Vec<_>>(), fresh);
        rows[0].zone = 9;
        stood
    };
    let (mut facts, cx) = (single(), Constraints::default());
    let mut own = Vec::new();
    for second in [false, true] {
        // The thread's whole write, by a decision: its rows stand from the
        // second on, and carry nothing of the caller's write between.
        let decided = decide(spec.engine(), &pool, &workers, &live, &mut facts, &cx);
        assert_eq!(picked(&decided).0, 0);
        assert_eq!(with_buffers(|buffers| stood(&mut buffers.whole)), second);
        // A caller's own buffer over the same list and snapshot, as a
        // pair's decision writes it: no row stands in it, so the pin, the
        // admission bits and the pair class written behind one write are
        // gone at the next.
        let (written, _) = pool.write(&workers, &live, &mut own);
        assert!(!stood(written));
        for row in written {
            row.session_hit = true;
            row.has_capacity = false;
            row.has_room = false;
            row.pair_class = 7;
        }
    }
}

#[test]
fn rows_stand_for_their_own_list_and_snapshot_alone() {
    let clock = DecisionClock::new();
    let spec = applied("least_load");
    let (north, south) = (fleet(3), fleet(5));
    south[0].set_status(WorkerStatus::NotReady);
    let pools = [
        PoolRecords::new(&north, &spec, &clock),
        PoolRecords::new(&south, &spec, &clock),
        // A membership change: new records, here of the same workers.
        PoolRecords::new(&north, &spec, &clock),
    ];
    let lists = [&north, &south, &north];
    let (first, _) = reports(&[10, 20, 30, 40, 50], Duration::from_millis(5));
    let (second, _) = reports(&[50, 40, 30, 20, 10], Duration::from_millis(5));
    let mut at = Instant::now();
    let (mut rows, mut stands) = (Vec::new(), Standing::default());
    // Each write marks a field no write changes: where the mark is read
    // again the rows stood, and nothing was copied. Each decision is later
    // than the one before, so a new snapshot's reports are converted.
    let mut stood = |list: usize, loads: Option<&Arc<LoadSnapshot>>| {
        at += Duration::from_micros(1);
        let live = Live::at(spec.engine(), &clock, at, loads, None);
        let (written, _) =
            pools[list].write_over::<true, false>(lists[list], &live, &mut rows, &mut stands);
        let stood = mem::replace(&mut written[0].zone, 0) == 9;
        let texts: Vec<_> = written.iter().map(text).collect();
        assert_eq!(texts, fresh(&pools[list], lists[list], &live));
        written[0].zone = 9;
        stood
    };
    // One list under one snapshot: its rows stand from the second write on.
    assert_eq!(
        [stood(0, Some(&first)), stood(0, Some(&first))],
        [false, true]
    );
    // Two lists in turn, new records of the same workers, a new snapshot,
    // and no snapshot at all: each brings its own rows.
    for (list, loads) in [
        (1, Some(&first)),
        (0, Some(&first)),
        (2, Some(&first)),
        (2, Some(&second)),
        (2, None),
    ] {
        assert!(!stood(list, loads), "list {list}");
        assert!(stood(list, loads), "list {list}");
    }
}

#[test]
fn a_decision_asked_inside_a_decision_leaves_its_rows_alone() {
    let clock = DecisionClock::new();
    let spec = applied("least_load");
    let (outer, inner) = (fleet(3), fleet(2));
    let pools = [
        PoolRecords::new(&outer, &spec, &clock),
        PoolRecords::new(&inner, &spec, &clock),
    ];
    let (loads, _) = reports(&[30, 20, 10], Duration::from_millis(5));
    let live = Live::at(spec.engine(), &clock, Instant::now(), Some(&loads), None);
    with_buffers(|buffers| {
        let Buffers { whole, stands, .. } = buffers;
        let (written, _) = pools[0].write_over::<true, false>(&outer, &live, whole, stands);
        let before: Vec<_> = written.iter().map(text).collect();
        let named = *stands;
        // A stage that asks a decision of its own on this thread, over
        // another list: it takes buffers of its own.
        let mut facts = single();
        let cx = Constraints::default();
        let asked = decide(spec.engine(), &pools[1], &inner, &live, &mut facts, &cx);
        assert_eq!(picked(&asked).0, 1);
        assert_eq!(whole.iter().map(text).collect::<Vec<_>>(), before);
        assert_eq!(*stands, named);
    });
}
