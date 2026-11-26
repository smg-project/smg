use super::*;

/// A session of the table's own cases, by its value.
fn s(value: u64) -> Session {
    Session::from_raw(value)
}

const RULE: SessionRule = SessionRule {
    assignment: Assignment::Delegate,
    inflight_cap: 2,
    ttl_micros: Some(1_000),
    sweep_micros: 100,
    recovery: Recovery::Latest,
    rule_placed: RulePlaced::Told,
};

const ORIGINAL: SessionRule = SessionRule {
    recovery: Recovery::Original,
    ..RULE
};

#[test]
fn the_latest_dispatch_leads_and_the_one_before_it_is_the_failover() {
    let table = SessionTable::new(RULE.sweep_micros);
    assert_eq!(table.lookup(s(7), 10, &RULE), [WorkerId(0); 2]);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(0)]));
    // A hit on the preferred pin changes nothing.
    table.lookup(s(7), 20, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(0)]));
    // A worker outside the pins (the pins were unroutable) leads.
    table.lookup(s(7), 30, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(2));
    assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
    // The failover taken while the preferred one is unroutable keeps the
    // order; a respill to it makes it the preferred one.
    table.lookup(s(7), 40, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
    table.lookup(s(7), 50, &RULE);
    table.dispatched(s(7), SessionVerdict::Respill, WorkerId(1));
    assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(2)]));
}

#[test]
fn under_original_recovery_the_first_worker_keeps_the_lead() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(3), 10, &ORIGINAL);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(1));
    table.lookup(s(3), 20, &ORIGINAL);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2));
    assert_eq!(table.pins(s(3)), Some([WorkerId(1), WorkerId(2)]));
    // A third worker evicts the oldest.
    table.lookup(s(3), 30, &ORIGINAL);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3));
    assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(3)]));
}

#[test]
fn idle_keys_are_swept_once_the_interval_passes() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 0, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    table.lookup(s(2), 900, &RULE);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2));
    assert_eq!(table.len(), 2);
    // Within the sweep interval nothing is swept.
    table.lookup(s(5), 950, &RULE);
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    // Past the interval and the first key's TTL: the first key is gone,
    // the second (touched at 900) stays.
    table.lookup(s(6), 1_100, &RULE);
    assert_eq!(table.pins(s(1)), None);
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
    // Without a TTL nothing expires.
    let forever = SessionTable::new(0);
    let no_ttl = SessionRule {
        ttl_micros: None,
        ..RULE
    };
    forever.lookup(s(1), 0, &no_ttl);
    forever.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    forever.lookup(s(2), 1_000_000, &no_ttl);
    assert_eq!(forever.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
}

/// Two rules share one table: the sweep runs at the shortest interval
/// among them, so a rule with an hour between sweeps never defers the
/// expiry of another rule's keys.
#[test]
fn a_long_sweep_rule_does_not_defer_another_rules_expiry() {
    let hourly = SessionRule {
        ttl_micros: Some(3_600_000_000),
        sweep_micros: 3_600_000_000,
        ..RULE
    };
    let table = SessionTable::new(RULE.sweep_micros.min(hourly.sweep_micros));
    table.lookup(s(1), 0, &hourly);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    table.lookup(s(2), 0, &RULE);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2));
    // Past the short rule's TTL the next decision sweeps its key; the
    // hourly key stays.
    assert_eq!(table.lookup(s(2), 2_000, &RULE), [WorkerId(0); 2]);
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
}

/// A table that outlives a document takes the next one's two times and
/// keeps its keys. A table that never swept takes an interval, and a
/// pass is due at once. A document with no interval leaves the one
/// that stands, so the keys the document before wrote still leave at a
/// pass. A shorter interval waits for no pass of the longer one.
#[test]
fn a_table_takes_a_new_documents_two_times_and_keeps_its_keys() {
    let pin = |table: &SessionTable, key: u64, at: u64| {
        table.lookup(mixed(key), at, &RULE);
        table.dispatched(mixed(key), SessionVerdict::Pinned, WorkerId(key));
    };
    // A table made with no interval runs no pass over its idle keys
    // ahead of the clock's last instant: a key past its own rule's end
    // stands at a timer's call.
    let table = SessionTable::new(0).with_shards(8);
    pin(&table, 1, 0);
    table.sweep_due(5_000);
    assert_eq!(table.len(), 1, "a table with no interval sweeps nothing");
    // The next document's rule sweeps: the key past its end leaves at
    // the pass that is due now.
    table.set_times(RULE.sweep_micros, 7_000);
    assert_eq!((table.sweep_micros(), table.absence_micros()), (100, 7_000));
    assert_eq!(table.len(), 1, "the new times keep the keys");
    table.sweep_due(5_000);
    assert_eq!(table.len(), 0, "the first pass under the new interval");
    // A key of that document, then a document with no interval: the
    // interval stands, and the key leaves at its pass.
    pin(&table, 2, 5_000);
    table.set_times(0, 9_000);
    assert_eq!((table.sweep_micros(), table.absence_micros()), (100, 9_000));
    assert_eq!(table.len(), 1, "the new times keep the keys");
    let pins = table.pins(mixed(2));
    assert_eq!(pins, Some([WorkerId(2), WorkerId(0)]), "and their pins");
    table.sweep_due(6_200);
    assert_eq!(table.len(), 0, "a pass under the interval that stood");
    // New times that name the interval that stands move no instant: the
    // pass just taken is not due again.
    assert_eq!(instants(&table), (6_300, 6_300));
    table.set_times(RULE.sweep_micros, 9_500);
    assert_eq!(table.absence_micros(), 9_500);
    assert_eq!(instants(&table), (6_300, 6_300), "the same interval");

    // A long interval, its first pass behind it, and a key past its end.
    let table = SessionTable::new(1_000_000).with_shards(8);
    table.sweep_due(10);
    pin(&table, 3, 20);
    table.sweep_due(2_000);
    assert_eq!(table.len(), 1, "the long interval's pass is not due");
    table.set_times(RULE.sweep_micros, 7_000);
    assert_eq!(table.len(), 1, "the new times keep the keys");
    table.sweep_due(2_000);
    assert_eq!(table.len(), 0, "the shorter interval's pass is due at once");
}

/// The table's two instants: its next sweep, and its next pass over the
/// idle keys.
fn instants(table: &SessionTable) -> (u64, u64) {
    let read = |word: &AtomicU64| word.load(Ordering::Relaxed);
    (
        read(&table.next_sweep_micros),
        read(&table.next_pass_micros),
    )
}

/// A request's sweep claims its turn by taking the list's lock, and
/// where the lock is held it takes no turn and waits for none: the two
/// instants stand, the keys stand, and the sweep stays due. A later
/// call takes the turn the first one left.
///
/// Another thread holds the lock until the case has looked, or until it
/// has waited two minutes for that word. A sweep that waited for the
/// lock goes on behind that and takes its turn, so the case fails where
/// it would hang.
#[test]
fn a_requests_sweep_takes_no_turn_and_waits_for_none_while_the_lists_lock_is_held() {
    use std::sync::{mpsc, Arc};
    let table = Arc::new(SessionTable::new(1_000_000).with_shards(8));
    // The table's first request sweeps: the next pass is one interval
    // on. Its key ends at 1,100.
    table.lookup(mixed(4), 10, &RULE);
    assert_eq!(instants(&table), (1_000_010, 1_000_010));
    let (held, is_held) = mpsc::channel();
    let (looked, has_looked) = mpsc::channel::<()>();
    let holder = {
        let table = Arc::clone(&table);
        std::thread::spawn(move || {
            let absent = table.absent.lock();
            held.send(()).expect("the case waits for the holder");
            let _ = has_looked.recv_timeout(std::time::Duration::from_secs(120));
            drop(absent);
        })
    };
    is_held.recv().expect("the holder took the lock");
    table.lookup(mixed(5), 1_000_020, &RULE);
    assert_eq!(instants(&table), (1_000_010, 1_000_010), "no turn taken");
    assert_eq!(table.len(), 2, "no turn taken: the key past its end stands");
    looked.send(()).expect("the holder waits for the case");
    holder.join().expect("the holder ends");
    table.lookup(mixed(6), 1_000_020, &RULE);
    assert_eq!(instants(&table), (2_000_020, 2_000_020), "the turn left");
    assert_eq!(table.len(), 2, "the key past its end left");
}

/// What a request read ahead of the lock is read again under it. A call
/// whose turn moved meanwhile counts nothing by its own instant. A
/// document's pass stays due, also behind a call that read a leave's
/// store, whose instant is the document's own. A leave is stamped by
/// the call that reads the leave's store, and a turn is taken once. A
/// request reads the interval ahead of the instant, so new times that
/// land between its two reads show as the interval before.
#[test]
fn a_requests_sweep_whose_turn_moved_ahead_of_the_lock_leaves_it() {
    const OLD: u64 = 1_000_000;
    let table = SessionTable::new(OLD).with_shards(8);
    table.lookup(mixed(4), 10, &RULE);
    table.dispatched(mixed(4), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(instants(&table), (1_000_010, 1_000_010));
    // A request read that turn as due and stalled. New times then made
    // a pass due at once.
    let new = RULE.sweep_micros;
    table.set_times(new, 7_000);
    assert_eq!(instants(&table), (0, 0));
    table.sweep_where_free((1_000_010, OLD), 1_000_015);
    assert_eq!(instants(&table), (0, 0), "the document's pass stays due");
    assert_eq!(table.len(), 1, "and no key was walked");
    // A request that read a leave's store ahead of the new times read
    // the instant that stands, and the interval before.
    table.sweep_where_free((0, OLD), 400);
    assert_eq!(instants(&table), (0, 0), "the document's pass stays due");
    // The request that reads both as they stand takes the turn.
    table.sweep_where_free((0, new), 500);
    assert_eq!(instants(&table), (600, 600));
    // A leave stores its own instant. A request that read the one
    // before stamps nothing: the absence is timed from the call that
    // reads the leave's store.
    table.left(WorkerId(1));
    assert_eq!(instants(&table), (0, 600));
    table.sweep_where_free((600, new), 650);
    assert_eq!(*table.absent.lock(), [(WorkerId(1), 0)], "not stamped");
    table.sweep_where_free((0, new), 700);
    assert_eq!(*table.absent.lock(), [(WorkerId(1), 7_700)]);
    assert_eq!(instants(&table), (800, 800));
    // A second request read the leave's store too, and runs behind the
    // one that took the turn.
    table.sweep_where_free((0, new), 900);
    assert_eq!(instants(&table), (800, 800), "a turn is taken once");
    // A request's own two reads, taken as they stand: a call dated
    // ahead of the turn finds nothing due under the lock, and the call
    // at 900 takes the turn.
    table.sweep_as_read(750);
    assert_eq!(instants(&table), (800, 800), "nothing is due at 750");
    table.sweep_as_read(900);
    assert_eq!(instants(&table), (1_000, 1_000));
}

/// A sweep asked off the path of a request waits for the list's lock,
/// and what is due is read under it. A leave that stores its instant
/// while the sweep waits is stamped by that sweep, and a pass another
/// call took while it waited is not walked twice. A document's new
/// times wait for the lock too, with both of them. Each call runs on a
/// thread that says it started, and the holder then yields to it. A
/// thread that has not reached the lock by then hides a call that did
/// not wait: it fails no call that does.
#[test]
fn a_timers_sweep_and_new_times_wait_for_the_lists_lock() {
    use std::sync::Arc;
    const LONG: u64 = 1_000_000;
    // What the lock's holder writes before it lets go.
    type Writes = fn(&SessionTable, &mut Vec<(WorkerId, u64)>);
    let table = Arc::new(SessionTable::new(LONG).with_shards(8));
    table.sweep_due(10);
    table.lookup(mixed(4), 20, &RULE);
    assert_eq!(instants(&table), (1_000_010, 1_000_010));
    // `call` on a thread of its own while this thread holds the lock:
    // whether it was still waiting, and the instants and the two times
    // that stood, all read ahead of the lock's release. The holder
    // `writes` under the lock before it lets go.
    let behind_the_lock = |call: fn(&SessionTable), writes: Writes| {
        let started = Arc::new(AtomicBool::new(false));
        let mut absent = table.absent.lock();
        let thread = {
            let (table, started) = (Arc::clone(&table), Arc::clone(&started));
            std::thread::spawn(move || {
                started.store(true, Ordering::Relaxed);
                call(&table);
            })
        };
        while !started.load(Ordering::Relaxed) {
            std::thread::yield_now();
        }
        for _ in 0..1_000 {
            std::thread::yield_now();
        }
        let stood = (
            !thread.is_finished(),
            instants(&table),
            (table.sweep_micros(), table.absence_micros()),
        );
        writes(&table, &mut absent);
        drop(absent);
        thread.join().expect("the call ends");
        stood
    };
    let times = (LONG, ABSENT_WORKER_PIN_TTL_MICROS);
    // A leave's two stores while the call waited: the call takes the
    // turn and stamps the leave.
    let leaves: Writes = |table, absent| {
        absent.push((WorkerId(9), 0));
        table.next_sweep_micros.store(0, Ordering::Relaxed);
    };
    let stood = behind_the_lock(|table| table.sweep_due(1_000_020), leaves);
    assert_eq!(stood, (true, (1_000_010, 1_000_010), times));
    assert_eq!(table.len(), 0, "the turn is taken behind the holder");
    let ends = 1_000_020 + ABSENT_WORKER_PIN_TTL_MICROS;
    assert_eq!(*table.absent.lock(), [(WorkerId(9), ends)], "the leave too");
    assert_eq!(instants(&table), (2_000_020, 2_000_020));
    // The pass another call took while this one waited, as that call's
    // two stores: this one walks nothing, and a key past its end stands.
    table.lookup(mixed(5), 2_000_030, &RULE);
    let taken: Writes = |table, _| {
        table.next_pass_micros.store(4_000_000, Ordering::Relaxed);
        table.next_sweep_micros.store(4_000_000, Ordering::Relaxed);
    };
    let stood = behind_the_lock(|table| table.sweep_due(3_000_000), taken);
    assert_eq!(stood, (true, (2_000_020, 2_000_020), times));
    assert_eq!(table.len(), 1, "a pass is not walked twice");
    assert_eq!(instants(&table), (4_000_000, 4_000_000));
    // New times wait for the lock with both of them.
    let stood = behind_the_lock(|table| table.set_times(RULE.sweep_micros, 7_000), |_, _| {});
    assert_eq!(stood, (true, (4_000_000, 4_000_000), times));
    assert_eq!(instants(&table), (0, 0), "the new interval's pass is due");
    let times = (table.sweep_micros(), table.absence_micros());
    assert_eq!(times, (RULE.sweep_micros, 7_000));
}

/// A key renewed under a rule of another sweep interval keeps the end
/// the interval before gave it, while that end is less than one new
/// interval past the end of the key's idle time: the end stays at or
/// after the end of that idle time. Past the span the new interval's
/// own sweep replaces the end.
#[test]
fn a_key_renewed_under_a_new_interval_keeps_an_end_inside_it() {
    const SECOND: u64 = 1_000_000;
    let rule = |sweep_s: u64| SessionRule {
        ttl_micros: Some(10 * SECOND),
        sweep_micros: sweep_s * SECOND,
        ..RULE
    };
    let ends = |table: &SessionTable| table.held(s(1)).map(|(_, ends)| ends);
    let table = SessionTable::new(60 * SECOND);
    // Idle until 45 s: the first sweep of 60 s at or after it is at 60 s.
    table.lookup(s(1), 35 * SECOND, &rule(60));
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(ends(&table), Some(SessionExpiry::At(60 * SECOND)));
    // The next document's rule sweeps every 40 s. A use at 45 s is idle
    // until 55 s: the end at 60 s is less than 40 s past it and stands,
    // where the new interval alone gives 80 s.
    table.lookup(s(1), 45 * SECOND, &rule(40));
    assert_eq!(ends(&table), Some(SessionExpiry::At(60 * SECOND)));
    // A use at 58 s is idle until 68 s, past the old end: the new
    // interval's sweep at 80 s replaces it.
    table.lookup(s(1), 58 * SECOND, &rule(40));
    assert_eq!(ends(&table), Some(SessionExpiry::At(80 * SECOND)));
}

/// A host reads a key's pins with the instant its rule forgets it, and
/// no instant under a rule without a TTL. A key a dispatch alone wrote
/// is held unread.
#[test]
fn a_key_is_held_with_the_instant_its_rule_forgets_it() {
    let table = SessionTable::new(RULE.sweep_micros);
    assert_eq!(table.held(s(1)), None);
    table.lookup(s(1), 50, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    // Idle until 1 050: the rule's next sweep is at 1 100.
    assert_eq!(
        table.held(s(1)),
        Some(([WorkerId(1), WorkerId(0)], SessionExpiry::At(1_100)))
    );
    // A read past the idle time and ahead of that sweep renews the key.
    assert_eq!(table.lookup(s(1), 1_060, &RULE), [WorkerId(1), WorkerId(0)]);
    let renewed = table.held(s(1)).map(|(_, ends)| ends);
    assert_eq!(renewed, Some(SessionExpiry::At(2_100)));
    let for_ever = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..RULE
    };
    table.lookup(s(2), 50, &for_ever);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2));
    assert_eq!(
        table.held(s(2)),
        Some(([WorkerId(2), WorkerId(0)], SessionExpiry::Never))
    );
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3));
    assert_eq!(
        table.held(s(3)),
        Some(([WorkerId(3), WorkerId(0)], SessionExpiry::Unread))
    );
}

/// The read with nothing written gives a key's pins as a decision at
/// that instant finds them: none for a key the table never read, the
/// pins until the key's end, and none from that instant. It makes no
/// key, renews nothing and forgets nothing.
#[test]
fn a_keys_pins_are_read_as_they_stand_with_nothing_written() {
    let table = SessionTable::new(RULE.sweep_micros);
    assert_eq!(table.pins_at(s(1), 50), [WorkerId(0); 2]);
    assert!(table.is_empty(), "the read makes no key");
    table.lookup(s(1), 50, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(7));
    let held = table.held(s(1));
    assert_eq!(table.pins_at(s(1), 1_099), [WorkerId(7), WorkerId(0)]);
    assert_eq!(table.pins_at(s(1), 1_100), [WorkerId(0); 2], "its end");
    assert_eq!(table.pins_at(s(1), u64::MAX), [WorkerId(0); 2]);
    assert_eq!(table.held(s(1)), held, "nothing renewed or forgotten");
    // Under a rule with no idle time the pins stand at every instant
    // but the clock's last, and a key dispatched to and never read has
    // none.
    let for_ever = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..RULE
    };
    table.lookup(s(2), 50, &for_ever);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(9));
    assert_eq!(
        table.pins_at(s(2), u64::MAX - 1),
        [WorkerId(9), WorkerId(0)]
    );
    assert_eq!(table.pins_at(s(2), u64::MAX), [WorkerId(0); 2]);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3));
    assert_eq!(table.pins_at(s(3), 50), [WorkerId(0); 2]);
}

/// A key past its end is forgotten at its next lookup even when no
/// sweep ran since: the compare is the key's own, the walk only reclaims.
#[test]
fn an_expired_key_is_forgotten_at_its_lookup_before_any_sweep() {
    let rarely = SessionTable::new(1_000_000);
    rarely.lookup(s(1), 0, &RULE);
    rarely.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(rarely.lookup(s(1), 500, &RULE), [WorkerId(1), WorkerId(0)]);
    assert_eq!(rarely.lookup(s(1), 2_000, &RULE), [WorkerId(0); 2]);
}

/// The rounding writes no end a host would misread. A rule with an idle
/// time and no sweep interval ends a key with its idle time. A sweep at
/// the clock's last instant, which reads as no end, is not taken.
#[test]
fn a_keys_end_is_never_rounded_to_no_end() {
    let table = SessionTable::new(RULE.sweep_micros);
    let ends = |key, now, sweep_micros| {
        let rule = SessionRule {
            ttl_micros: Some(60),
            sweep_micros,
            ..RULE
        };
        table.lookup(s(key), now, &rule);
        table.held(s(key)).map(|(_, ends)| ends)
    };
    assert_eq!(ends(1, 10, 0), Some(SessionExpiry::At(70)));
    // The clock's last instant is a multiple of five.
    assert_eq!(
        ends(2, u64::MAX - 63, 5),
        Some(SessionExpiry::At(u64::MAX - 3))
    );
}

/// One key read under two rules, as no engine reads one: a key meets one
/// rule while its table lives. An end another rule wrote stays where it
/// stands less than one interval above the new idle end, and one a whole
/// interval above is replaced by this rule's sweep.
#[test]
fn an_end_another_rule_wrote_stays_inside_one_interval_alone() {
    let table = SessionTable::new(RULE.sweep_micros);
    let ends = |key, now, ttl, sweep_micros| {
        let rule = SessionRule {
            ttl_micros: Some(ttl),
            sweep_micros,
            ..RULE
        };
        table.lookup(s(key), now, &rule);
        table.held(s(key)).map(|(_, ends)| ends)
    };
    // A sweep every 70 ends a read at 30 at 1 050. RULE's read at 40
    // ends the idle time at 1 040, and 1 050 stands inside one interval
    // of 100 above it: RULE alone gives 1 100.
    assert_eq!(ends(1, 30, 1_000, 70), Some(SessionExpiry::At(1_050)));
    assert_eq!(ends(1, 40, 1_000, 100), Some(SessionExpiry::At(1_050)));
    // An end exactly one interval above the new idle end is replaced.
    assert_eq!(ends(2, 0, 1_100, 0), Some(SessionExpiry::At(1_100)));
    assert_eq!(ends(2, 0, 1_000, 100), Some(SessionExpiry::At(1_000)));
}

/// The table against a model of the gateway's sticky map, over seeded
/// histories of one key. The map ends an entry at an eviction tick
/// alone, one tick an interval, where the entry's last use is the idle
/// time old, and every use renews the entry with no compare
/// (`model_gateway/src/policies/manual.rs:168-179`, `:395`, `:234`).
/// Each request finds its key kept or started over as the map does,
/// under reads that sweep and under a host's own pass. A model, not a
/// recorded line: no recorded row moves time.
#[test]
fn a_keys_pins_end_as_an_entry_of_the_gateways_sticky_map_does() {
    let mut seed = 7_u64;
    let mut draw = |bound: u64| {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) % bound
    };
    let (mut requests, mut differ) = (0_u32, 0_u32);
    for history in 0..20_000 {
        let idle = [1, 7, 30, 60, 61, 100, 240, 1_000][draw(8) as usize];
        let interval = [1, 7, 60, 100, 3_600][draw(5) as usize];
        let rule = SessionRule {
            ttl_micros: Some(idle),
            sweep_micros: interval,
            ..RULE
        };
        let table = SessionTable::new(interval).with_shards(1);
        // The map's one entry: the instant of its last use while it is
        // kept, and the map's next tick.
        let (mut used, mut tick) = (None, interval);
        let mut now = 0;
        for _ in 0..2 + draw(39) {
            let long = 1 + draw(2 * idle + 2 * interval);
            now += [0, 1, long, 1 + draw(idle + 1)][draw(4) as usize];
            if history % 2 == 1 {
                table.sweep_due(now);
            }
            let kept = table.lookup(s(1), now, &rule) != [WorkerId(0); 2];
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
            // A tick at a use's instant runs ahead of the use.
            while tick <= now {
                if used.is_some_and(|used| tick - used >= idle) {
                    used = None;
                }
                tick += interval;
            }
            differ += u32::from(kept != used.replace(now).is_some());
            requests += 1;
        }
    }
    assert_eq!((differ, requests), (0, 420_215), "answers that differ");
}

#[test]
fn an_older_lookup_cannot_shorten_a_newer_touch() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(7), 500, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));

    table.lookup(s(7), 1_000, &RULE);
    table.lookup(s(7), 600, &ORIGINAL);

    assert_eq!(
        table.held(s(7)),
        Some(([WorkerId(1), WorkerId(0)], SessionExpiry::At(2_000)))
    );
    assert_eq!(
        table.claim(s(7), [WorkerId(1), WorkerId(0)], WorkerId(2)),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.lookup(s(7), 1_600, &RULE), [WorkerId(2), WorkerId(1)]);
}

/// A retry's claim survives the commit of an older request on the owner:
/// only the claiming decision settles its claim, so the retries after it
/// read the claim as the pin of record instead of claiming elsewhere, and
/// the claim is confirmed when its own commit lands.
#[test]
fn an_older_requests_commit_on_the_owner_leaves_a_retrys_claim_standing() {
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 0, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(1)),
        Claim::Made(WorkerId(1))
    );
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(1));
    // An older request decided on the owner is still in flight when a
    // retry that cannot use the owner claims B.
    let seen = table.lookup(s(1), 10, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(2)),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(2)]),
        "the older request's commit on the owner leaves the claim standing"
    );
    // The next retry reads B as the pin of record; the claiming retry's
    // own commit confirms it.
    assert_eq!(
        table.lookup(s(1), 20, &ORIGINAL),
        [WorkerId(1), WorkerId(2)]
    );
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(2));
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
}

#[test]
fn an_empty_key_or_target_writes_nothing() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.dispatched(s(0), SessionVerdict::Pinned, WorkerId(1));
    table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(0));
    assert!(table.is_empty());
}

/// A respill's move, written ahead of its commit, fronts the worker
/// taken under either recovery, as the commit does. It makes no entry:
/// an empty key, an empty worker and a key the table holds no more
/// write nothing.
#[test]
fn a_respills_move_fronts_the_worker_taken_and_makes_no_entry() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(7), 10, &rule);
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
        table.respilled(s(7), WorkerId(2));
        assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
        // The worker that leads already: nothing moves, and the commit
        // finds the pins in place.
        table.respilled(s(7), WorkerId(2));
        table.dispatched(s(7), SessionVerdict::Respill, WorkerId(2));
        assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
        table.respilled(s(0), WorkerId(3));
        table.respilled(s(7), WorkerId(0));
        table.respilled(s(8), WorkerId(3));
        assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
        assert_eq!((table.pins(s(8)), table.len()), (None, 1));
    }
}

/// Two decisions read one pin at its cap, and each takes another
/// worker. Both front, the later one last, and their commits front in
/// their own order: the pins end where the two commits alone leave
/// them.
#[test]
fn two_respills_of_one_capped_pin_both_front_the_later_one_last() {
    let ends = [true, false].map(|moved| {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(7), 10, &RULE);
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
        // Both decisions read the pin ahead of either's move.
        let read = [20, 21].map(|now| table.lookup(s(7), now, &RULE));
        assert_eq!(read, [[WorkerId(1), WorkerId(0)]; 2]);
        if moved {
            table.respilled(s(7), WorkerId(2));
            table.respilled(s(7), WorkerId(3));
            assert_eq!(table.pins(s(7)), Some([WorkerId(3), WorkerId(2)]));
        }
        // The later decision's request is sent first.
        table.dispatched(s(7), SessionVerdict::Respill, WorkerId(3));
        table.dispatched(s(7), SessionVerdict::Respill, WorkerId(2));
        table.pins(s(7))
    });
    assert_eq!(ends, [Some([WorkerId(2), WorkerId(3)]); 2]);
}

/// A respill's move beside a claim no commit has confirmed leaves the
/// claim to its own commit. On the claimed worker that commit confirms
/// it, and elsewhere it withdraws it. Whichever of the two commits
/// comes first, under either recovery, the pins end where the two
/// commits alone leave them.
#[test]
fn a_respills_move_beside_an_open_claim_ends_where_the_commits_alone_end() {
    let (claimed, taken) = (WorkerId(1), WorkerId(2));
    for rule in [RULE, ORIGINAL] {
        for lands in [claimed, WorkerId(3)] {
            for claim_first in [true, false] {
                let ends = [true, false].map(|moved| {
                    let table = SessionTable::new(rule.sweep_micros);
                    let seen = table.lookup(s(1), 10, &rule);
                    assert_eq!(table.claim(s(1), seen, claimed), Claim::Made(claimed));
                    if moved {
                        table.respilled(s(1), taken);
                        assert_eq!(table.pins(s(1)), Some([taken, claimed]));
                    }
                    let claims = SessionVerdict::Claimed(claimed);
                    if claim_first {
                        table.dispatched(s(1), claims, lands);
                    }
                    table.dispatched(s(1), SessionVerdict::Respill, taken);
                    if !claim_first {
                        table.dispatched(s(1), claims, lands);
                    }
                    table.pins(s(1))
                });
                assert_eq!(
                    ends[0], ends[1],
                    "{:?}: the claim's commit on {lands:?}, first: {claim_first}",
                    rule.recovery
                );
                assert_eq!(ends[0].map(|pins| pins.contains(&taken)), Some(true));
            }
        }
    }
}

/// Two decisions on one key interleave: A reads, B reads, A notes its
/// respill, B dispatches onto the failover pin (a hit, its instant is
/// not the note's), then A dispatches onto it (the respill). The note is
/// keyed by the decision instant and consumed by its own dispatch alone.
#[test]
fn a_respill_verdict_travels_with_its_decision_past_another_decisions_reading() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(9), 10, &RULE);
    table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1));
    table.lookup(s(9), 20, &RULE);
    table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(2));
    assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(1)]));
    // A decides a respill onto the failover, B reads the key and takes
    // the failover as a plain hit, B commits first: B's hit leaves the
    // pins, A's respill fronts the failover whatever came between.
    table.lookup(s(9), 30, &RULE);
    table.lookup(s(9), 31, &RULE);
    table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(
        table.pins(s(9)),
        Some([WorkerId(2), WorkerId(1)]),
        "B's dispatch onto the failover is a hit"
    );
    table.dispatched(s(9), SessionVerdict::Respill, WorkerId(1));
    assert_eq!(
        table.pins(s(9)),
        Some([WorkerId(1), WorkerId(2)]),
        "A's dispatch is the respill"
    );
}

#[test]
fn the_first_claim_stands_for_a_fresh_key_and_for_a_key_whose_pins_are_down() {
    let table = SessionTable::new(RULE.sweep_micros);
    let none = [WorkerId(0); 2];
    let seen = table.lookup(s(5), 10, &ORIGINAL);
    assert_eq!(
        table.claim(s(5), seen, WorkerId(3)),
        Claim::Made(WorkerId(3)),
        "the first"
    );
    assert_eq!(
        table.claim(s(5), none, WorkerId(4)),
        Claim::Standing(WorkerId(3), [WorkerId(3), WorkerId(0)]),
        "follows it"
    );
    assert_eq!(table.pins(s(5)), Some([WorkerId(3), WorkerId(0)]));
    table.dispatched(s(5), SessionVerdict::Claimed(WorkerId(3)), WorkerId(3));
    table.dispatched(s(5), SessionVerdict::Pinned, WorkerId(4));
    assert_eq!(
        table.pins(s(5)),
        Some([WorkerId(3), WorkerId(4)]),
        "a loser dispatched elsewhere is the failover"
    );
    // Both pins down: two decisions read [3, 4]; the first claim appends
    // its worker by the recovery (the oldest slot goes), the second
    // follows the newcomer.
    let seen = table.lookup(s(5), 20, &ORIGINAL);
    assert_eq!(
        table.claim(s(5), seen, WorkerId(7)),
        Claim::Made(WorkerId(7))
    );
    assert_eq!(table.pins(s(5)), Some([WorkerId(4), WorkerId(7)]));
    assert_eq!(
        table.claim(s(5), seen, WorkerId(8)),
        Claim::Standing(WorkerId(7), [WorkerId(4), WorkerId(7)]),
        "the newcomer"
    );
    // Under the latest-first recovery the claim leads at once.
    let seen = table.lookup(s(6), 30, &RULE);
    assert_eq!(
        table.claim(s(6), seen, WorkerId(1)),
        Claim::Made(WorkerId(1))
    );
    assert_eq!(table.pins(s(6)), Some([WorkerId(1), WorkerId(0)]));
    assert_eq!(
        table.claim(s(0), none, WorkerId(9)),
        Claim::Standing(WorkerId(9), none),
        "no key, no claim"
    );
    assert_eq!(table.len(), 2);
}

/// Another request's respill turns the pins behind a decision's read:
/// the claim is not taken and no pin came in. The answer is the
/// caller's own worker with the pins the key held, and nothing is
/// recorded. Another decision that reads the same pins gets its own
/// worker: the answer is no pin to follow.
#[test]
fn a_claim_that_finds_no_new_pin_names_the_callers_own_worker() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(3), 10, &RULE);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(1));
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2));
    let seen = table.lookup(s(3), 20, &RULE);
    assert_eq!(seen, [WorkerId(2), WorkerId(1)]);
    table.dispatched(s(3), SessionVerdict::Respill, WorkerId(1));
    let held = [WorkerId(1), WorkerId(2)];
    assert_eq!(
        table.claim(s(3), seen, WorkerId(7)),
        Claim::Standing(WorkerId(7), held)
    );
    assert_eq!(table.pins(s(3)), Some(held), "nothing is recorded");
    assert_eq!(
        Claim::standing(&held, &seen, WorkerId(8)),
        Some(WorkerId(8))
    );
    assert_eq!(Claim::standing(&seen, &seen, WorkerId(8)), None);
}

/// The owner is down, the decision claims B, the commit refuses B and
/// dispatches the alternate C: the claim is withdrawn and the owner keeps
/// its slot (B served nothing); a dispatch to the claimed worker confirms
/// the claim instead, and the normal rule follows after it.
#[test]
fn a_claim_whose_commit_landed_elsewhere_is_withdrawn_before_the_pins_move() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    let seen = table.lookup(s(1), 20, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(2)),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3));
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(3)]),
        "the owner keeps the lead, the alternate is the failover"
    );
    // Under the latest-first recovery the claim led; the alternate leads
    // instead and the owner stays the failover.
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(2), 10, &RULE);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(1));
    let seen = table.lookup(s(2), 20, &RULE);
    assert_eq!(
        table.claim(s(2), seen, WorkerId(2)),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(1)]));
    table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3));
    assert_eq!(table.pins(s(2)), Some([WorkerId(3), WorkerId(1)]));
    // A dispatch to the claimed worker confirms the claim.
    let seen = table.lookup(s(2), 30, &RULE);
    assert_eq!(
        table.claim(s(2), seen, WorkerId(4)),
        Claim::Made(WorkerId(4))
    );
    table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(4)), WorkerId(4));
    assert_eq!(table.pins(s(2)), Some([WorkerId(4), WorkerId(3)]));
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(5));
    assert_eq!(
        table.pins(s(2)),
        Some([WorkerId(5), WorkerId(4)]),
        "a confirmed claim served: the normal rule moves it"
    );
}

#[test]
fn a_pin_another_request_already_used_survives_the_claimants_commit_elsewhere() {
    // X claims A for a fresh key; Y finds A pinned and commits there; A
    // goes unroutable and X's own commit lands on B. A was served, so it
    // stays: [A, B], and the recovered A keeps the key.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 0, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(1)),
        Claim::Made(WorkerId(1))
    );
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(2)]),
        "a pin another request already used survives the claimant's commit elsewhere"
    );
    assert_eq!(
        table.lookup(s(1), 10, &ORIGINAL),
        [WorkerId(1), WorkerId(2)]
    );
    // Under the latest-first recovery the alternate leads and A stays the
    // failover instead of vanishing.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(2), 0, &RULE);
    assert_eq!(
        table.claim(s(2), seen, WorkerId(1)),
        Claim::Made(WorkerId(1))
    );
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(1));
    table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(1)]));
}

#[test]
fn a_commit_elsewhere_by_another_request_leaves_the_claim_unconfirmed() {
    // The owner is pinned; a retry claims C; an older request commits on
    // the owner (not on C): the claim stands unconfirmed, so the retry's
    // own commit on D still withdraws C.
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    let seen = table.lookup(s(1), 20, &ORIGINAL);
    assert_eq!(
        table.claim(s(1), seen, WorkerId(2)),
        Claim::Made(WorkerId(2))
    );
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3));
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(3)]),
        "a commit elsewhere by another request confirms nothing"
    );
}

/// A decision that does not go to the worker it claimed moves the
/// claim while it stands unconfirmed: the worker claimed leaves the
/// pins, and the worker the request goes to is the claim in its place,
/// by the key's recovery. With no worker to go to, the claim is
/// withdrawn alone and the key is claimed afresh. A claim a dispatch
/// confirmed stands, and a worker that is a pin already takes no claim.
#[test]
fn a_claim_moves_to_the_worker_its_decision_goes_to_while_it_stands_unconfirmed() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let cases = [
        (&RULE, [2, 1], [3, 1], [4, 1]),
        (&ORIGINAL, [1, 2], [1, 3], [1, 4]),
    ];
    for (rule, claimed, moved, committed) in cases {
        // The owner is down: the decision claims 2, then goes to 3.
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 10, rule);
        table.dispatched(s(1), pinned, id(1));
        let seen = table.lookup(s(1), 20, rule);
        assert_eq!(table.claim(s(1), seen, id(2)), Claim::Made(id(2)));
        // A decision whose claim is not the one of record moves none.
        assert_eq!(table.move_claim(s(1), id(9), id(3)), pinned);
        assert_eq!(table.pins(s(1)), Some(claimed.map(id)));
        let moved_to = SessionVerdict::Claimed(id(3));
        assert_eq!(table.move_claim(s(1), id(2), id(3)), moved_to);
        assert_eq!(table.pins(s(1)), Some(moved.map(id)));
        // The worker it moved to is the claim: the decision's commit
        // elsewhere withdraws that one.
        table.dispatched(s(1), SessionVerdict::Claimed(id(3)), id(4));
        assert_eq!(table.pins(s(1)), Some(committed.map(id)));
        // No worker to go to: the claim is withdrawn alone, and the
        // key's next decision claims afresh.
        let seen = table.lookup(s(2), 10, rule);
        assert_eq!(table.claim(s(2), seen, id(2)), Claim::Made(id(2)));
        assert_eq!(table.move_claim(s(2), id(2), id(0)), pinned);
        assert_eq!(table.pins(s(2)), Some([id(0); 2]));
        assert_eq!(table.claim(s(2), seen, id(5)), Claim::Made(id(5)));
        // Another request was served on the claim: it stands.
        table.dispatched(s(2), pinned, id(5));
        assert_eq!(table.move_claim(s(2), id(5), id(6)), pinned);
        assert_eq!(table.pins(s(2)), Some([id(5), id(0)]));
        // A key the table does not hold is not written.
        assert_eq!(table.move_claim(s(3), id(1), id(2)), pinned);
        assert_eq!(table.len(), 2);
    }
    // Another request's commit made 3 a pin behind the claim on 2: the
    // claim leaves, and 3 stands as the pin it is, marked no claim. A
    // claimant's commit elsewhere then withdraws nothing.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &RULE);
    assert_eq!(table.claim(s(1), seen, id(2)), Claim::Made(id(2)));
    table.dispatched(s(1), pinned, id(3));
    assert_eq!(table.pins(s(1)), Some([id(3), id(2)]));
    assert_eq!(table.move_claim(s(1), id(2), id(3)), pinned);
    assert_eq!(table.pins(s(1)), Some([id(3), id(0)]));
    table.dispatched(s(1), SessionVerdict::Claimed(id(3)), id(9));
    assert_eq!(table.pins(s(1)), Some([id(9), id(3)]));
    // Two other requests' commits pushed the claim on 2 out of the
    // pins: withdrawn to no worker, it leaves them as they stand.
    let table = SessionTable::new(RULE.sweep_micros);
    let seen = table.lookup(s(1), 10, &ORIGINAL);
    assert_eq!(table.claim(s(1), seen, id(2)), Claim::Made(id(2)));
    table.dispatched(s(1), pinned, id(3));
    table.dispatched(s(1), pinned, id(4));
    assert_eq!(table.pins(s(1)), Some([id(3), id(4)]));
    assert_eq!(table.move_claim(s(1), id(2), id(0)), pinned);
    assert_eq!(table.pins(s(1)), Some([id(3), id(4)]));
}

#[test]
fn a_forgotten_worker_leaves_every_pin_and_a_key_with_no_pin_left_leaves_the_table() {
    let table = SessionTable::new(RULE.sweep_micros);
    for (key, first, second) in [(1, 1, 2), (2, 2, 0), (3, 2, 1), (4, 3, 0)] {
        table.lookup(s(key), 10, &ORIGINAL);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(first));
        if second != 0 {
            table.lookup(s(key), 20, &ORIGINAL);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(second));
        }
    }
    assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(1)]));
    table.forget_workers(&[WorkerId(2)]);
    assert_eq!(
        table.pins(s(1)),
        Some([WorkerId(1), WorkerId(0)]),
        "the failover slot went"
    );
    assert_eq!(table.pins(s(2)), None, "its only pin went");
    assert_eq!(
        table.pins(s(3)),
        Some([WorkerId(1), WorkerId(0)]),
        "the failover is the owner"
    );
    assert_eq!(
        table.pins(s(4)),
        Some([WorkerId(3), WorkerId(0)]),
        "untouched"
    );
    assert_eq!(table.len(), 3);
    table.forget_workers(&[WorkerId(0)]);
    assert_eq!(table.len(), 3, "the empty id is never a pin");
}

/// The pass over the idle keys reclaims what expired alone. A key a
/// decision read and has not claimed yet holds no pin, and stays.
#[test]
fn the_pass_keeps_a_key_read_and_not_yet_claimed() {
    let rule = SessionRule {
        ttl_micros: Some(100),
        sweep_micros: 1_000,
        ..ORIGINAL
    };
    let table = SessionTable::new(rule.sweep_micros);
    // The first pass: the next is due at 1,010.
    table.lookup(s(1), 10, &rule);
    assert_eq!(table.lookup(s(1), 1_005, &rule), [WorkerId(0); 2]);
    // Another key's decision runs the pass.
    table.lookup(s(2), 1_010, &rule);
    let claim = table.claim(s(1), [WorkerId(0); 2], WorkerId(3));
    assert_eq!(claim, Claim::Made(WorkerId(3)));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(3)), WorkerId(3));
    assert_eq!(table.lookup(s(1), 1_020, &rule), [WorkerId(3), WorkerId(0)]);
}

/// A walk that forgets a worker takes a key with no pin left out of
/// the table, and leaves the key a decision read and has not claimed.
#[test]
fn a_forgotten_workers_walk_keeps_a_key_read_and_not_yet_claimed() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &RULE);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(4));
    table.forget_workers(&[WorkerId(0)]);
    table.forget_workers(&[WorkerId(4)]);
    assert_eq!(table.pins(s(1)), Some([WorkerId(0); 2]));
    assert_eq!(table.pins(s(2)), None);
}

/// A claim follows its worker to its new id and keeps the id its lease
/// names. The claiming decision's own dispatch, which names the old
/// id, confirms the renamed claim. Its commit elsewhere withdraws it,
/// and the worker that served the key is pinned alone.
#[test]
fn a_claim_renamed_with_its_worker_is_confirmed_or_withdrawn_by_its_lease() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    for key in [7, 8] {
        assert_eq!(table.lookup(s(key), 10, &ORIGINAL), [WorkerId(0); 2]);
        let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(1));
        assert_eq!(claim, Claim::Made(WorkerId(1)));
    }
    table.rename_worker(WorkerId(1), WorkerId(11));
    let renamed = Some([WorkerId(11), WorkerId(0)]);
    assert_eq!((table.pins(s(7)), table.pins(s(8))), (renamed, renamed));
    // Its own dispatch names the id the worker had: confirmed.
    table.dispatched(s(7), SessionVerdict::Claimed(WorkerId(1)), WorkerId(1));
    assert_eq!(table.pins(s(7)), renamed);
    // A pin a dispatch confirmed is withdrawn by no later commit.
    table.dispatched(s(7), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
    assert_eq!(table.pins(s(7)), Some([WorkerId(11), WorkerId(2)]));
    // The other key's commit lands elsewhere: withdrawn.
    table.dispatched(s(8), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
    assert_eq!(table.pins(s(8)), Some([WorkerId(2), WorkerId(0)]));
}

/// A dispatch that lands behind its worker's rename names the id the
/// worker had. The pins move under the id that stands, through a
/// second rename too, and no pin names an id no row holds.
#[test]
fn a_dispatch_on_the_id_a_worker_had_moves_the_pins_under_the_id_that_stands() {
    let table = SessionTable::new(RULE.sweep_micros);
    for (key, worker) in [(1, 1), (2, 2)] {
        table.lookup(s(key), 10, &RULE);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
    }
    table.rename_worker(WorkerId(1), WorkerId(11));
    // The key's own late dispatch: its pin stands as renamed.
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
    // Another key's respill landed there: the standing id leads.
    table.dispatched(s(2), SessionVerdict::Respill, WorkerId(1));
    assert_eq!(table.pins(s(2)), Some([WorkerId(11), WorkerId(2)]));
    table.rename_worker(WorkerId(11), WorkerId(21));
    table.dispatched(s(1), SessionVerdict::Respill, WorkerId(1));
    assert_eq!(table.pins(s(1)), Some([WorkerId(21), WorkerId(0)]));
    // Back under the id it had first: that id stands again.
    table.rename_worker(WorkerId(21), WorkerId(1));
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(11));
    assert_eq!(table.pins(s(2)), Some([WorkerId(1), WorkerId(2)]));
    assert_eq!(table.renamed(), 2, "the two ids it left behind");
}

/// A decision reads its rows ahead of its move, and a worker may be
/// registered again between the two. The move fronts the id that
/// stands where the rename is over and where its walk has not reached
/// the key. A pin the walk has yet to rename leads under the id it
/// had, and the walk renames it there, as it renames a move made
/// ahead of the whole rename. The dispatch on the id the worker had
/// finds the pins in place, and no pin names an id no row holds.
#[test]
fn a_respills_move_on_the_id_a_worker_had_fronts_the_id_that_stands() {
    let table = SessionTable::new(RULE.sweep_micros);
    for key in [1, 2, 4] {
        table.lookup(s(key), 10, &RULE);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(2));
    }
    table.lookup(s(3), 10, &RULE);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(4));
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2));
    assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(4)]));
    // The whole rename is ahead of the move.
    table.rename_worker(WorkerId(1), WorkerId(11));
    table.respilled(s(1), WorkerId(1));
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(2)]));
    // The rename's first half: remembered, the shards not yet walked.
    table.remember(WorkerId(3), WorkerId(13));
    table.respilled(s(2), WorkerId(3));
    assert_eq!(table.pins(s(2)), Some([WorkerId(13), WorkerId(2)]));
    // The same half where the worker is the key's failover: its pin
    // leads as it stands, and the walk renames it.
    table.remember(WorkerId(4), WorkerId(14));
    table.respilled(s(3), WorkerId(4));
    assert_eq!(table.pins(s(3)), Some([WorkerId(4), WorkerId(2)]));
    // The move is ahead of the whole rename.
    table.respilled(s(4), WorkerId(5));
    assert_eq!(table.pins(s(4)), Some([WorkerId(5), WorkerId(2)]));
    for (had, stands) in [(3, 13), (4, 14), (5, 15)] {
        table.rename_worker(WorkerId(had), WorkerId(stands));
    }
    for (key, had, stands) in [(1, 1, 11), (2, 3, 13), (3, 4, 14), (4, 5, 15)] {
        let led = Some([WorkerId(stands), WorkerId(2)]);
        assert_eq!(table.pins(s(key)), led, "key {key} behind the renames");
        table.dispatched(s(key), SessionVerdict::Respill, WorkerId(had));
        assert_eq!(table.pins(s(key)), led, "key {key} behind its dispatch");
    }
}

/// A decision reads its rows ahead of its claim, and its worker may be
/// registered again between the two. The pin and the claim of record
/// take the id that stands, behind the whole rename and behind its
/// first half, and the answer keeps the id the decision read. The
/// lease names that id: the claim's own dispatch on it confirms the
/// claim, a later commit elsewhere withdraws no confirmed pin, and the
/// claiming decision's commit elsewhere withdraws its claim.
#[test]
fn a_claim_on_the_id_a_worker_had_pins_the_id_that_stands() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        // Worker 1 is registered again whole, worker 3 by the rename's
        // first half: remembered, the shards not yet walked.
        table.rename_worker(WorkerId(1), WorkerId(11));
        table.remember(WorkerId(3), WorkerId(13));
        let claims = [(1, 1, 11), (2, 1, 11), (3, 3, 13), (4, 3, 13)];
        for (key, had, stands) in claims {
            assert_eq!(table.lookup(s(key), 10, &rule), [WorkerId(0); 2]);
            let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(had));
            assert_eq!(
                claim,
                Claim::Made(WorkerId(had)),
                "the id the decision read"
            );
            let led = Some([WorkerId(stands), WorkerId(0)]);
            assert_eq!(
                table.pins(s(key)),
                led,
                "key {key} claims the id that stands"
            );
        }
        table.rename_worker(WorkerId(3), WorkerId(13));
        for (key, had, stands) in claims {
            let claimed = SessionVerdict::Claimed(WorkerId(had));
            if key % 2 == 1 {
                // Its own dispatch names the id the decision read.
                table.dispatched(s(key), claimed, WorkerId(had));
                let led = Some([WorkerId(stands), WorkerId(0)]);
                assert_eq!(table.pins(s(key)), led, "key {key} is confirmed");
                // A pin a dispatch confirmed is withdrawn by no commit.
                table.dispatched(s(key), claimed, WorkerId(2));
                let both = match rule.recovery {
                    Recovery::Latest => [WorkerId(2), WorkerId(stands)],
                    Recovery::Original => [WorkerId(stands), WorkerId(2)],
                };
                assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
            } else {
                // The claiming decision's commit lands elsewhere.
                table.dispatched(s(key), claimed, WorkerId(2));
                let alone = Some([WorkerId(2), WorkerId(0)]);
                assert_eq!(table.pins(s(key)), alone, "key {key} is withdrawn");
            }
        }
    }
}

/// A key pinned on a worker is claimed by a decision that read the
/// worker's row under the id it had: the key's pin, which names the id
/// that stands, read as not routable there, and the decision's pick is
/// the same worker. The table writes nothing and answers the
/// decision's own pick. A pin the key was served on is marked no
/// claim, so the request's commit on an alternate withdraws nothing.
#[test]
fn a_claim_on_a_worker_pinned_under_the_id_that_stands_writes_nothing() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        table.rename_worker(WorkerId(1), WorkerId(11));
        let pins = [WorkerId(11), WorkerId(0)];
        assert_eq!(table.lookup(s(1), 20, &rule), pins);
        let claim = table.claim(s(1), pins, WorkerId(1));
        assert_eq!(claim, Claim::Standing(WorkerId(1), pins));
        assert_eq!(table.pins(s(1)), Some(pins), "nothing is written");
        // The request's dispatch on its pick is one on the pin.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(table.pins(s(1)), Some(pins));
        // Its commit on an alternate moves the pins as a pin's does.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
        let both = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(11)],
            Recovery::Original => [WorkerId(11), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(both));
    }
}

/// A pair's decision moves its claim to the prefill its pair formed
/// on, which it read ahead of that write: a prefill registered again
/// between the two is claimed under the id that stands, behind the
/// whole rename and behind its first half, and the answer keeps the
/// id the decision read. Its dispatch on that id confirms the claim,
/// and its commit elsewhere withdraws it. A prefill that is a pin of
/// the key under the id that stands is no claim.
#[test]
fn a_moved_claim_on_the_id_a_worker_had_pins_the_id_that_stands() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        for key in [1, 2, 3] {
            table.lookup(s(key), 10, &rule);
            let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(5));
            assert_eq!(claim, Claim::Made(WorkerId(5)));
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        table.remember(WorkerId(3), WorkerId(13));
        for (key, had, stands) in [(1, 1, 11), (2, 1, 11), (3, 3, 13)] {
            let verdict = table.move_claim(s(key), WorkerId(5), WorkerId(had));
            let claimed = SessionVerdict::Claimed(WorkerId(had));
            assert_eq!(verdict, claimed, "the id the decision read");
            let led = Some([WorkerId(stands), WorkerId(0)]);
            assert_eq!(
                table.pins(s(key)),
                led,
                "key {key} claims the id that stands"
            );
        }
        table.rename_worker(WorkerId(3), WorkerId(13));
        for (key, had, stands) in [(1, 1, 11), (3, 3, 13)] {
            table.dispatched(
                s(key),
                SessionVerdict::Claimed(WorkerId(had)),
                WorkerId(had),
            );
            let led = Some([WorkerId(stands), WorkerId(0)]);
            assert_eq!(table.pins(s(key)), led, "key {key} is confirmed");
        }
        table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
        assert_eq!(
            table.pins(s(2)),
            Some([WorkerId(2), WorkerId(0)]),
            "withdrawn"
        );
        // The key was served on the prefill under the id that stands.
        table.lookup(s(4), 10, &rule);
        table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(11));
        let pins = [WorkerId(11), WorkerId(0)];
        assert_eq!(
            table.claim(s(4), pins, WorkerId(5)),
            Claim::Made(WorkerId(5))
        );
        let verdict = table.move_claim(s(4), WorkerId(5), WorkerId(1));
        assert_eq!(verdict, SessionVerdict::Pinned);
        assert_eq!(
            table.pins(s(4)),
            Some(pins),
            "the pin stands alone, and no claim"
        );
        table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(table.pins(s(4)), Some(pins));
    }
}

/// The race the table leaves as it stands: the key's own pin is
/// renamed between a decision's read of the key and its claim. The
/// pins are not the ones the decision read, and the renamed id answers
/// as their newcomer. No row of that decision holds it, so the request
/// keeps its own pick, and its commit moves the pins by the key's
/// recovery beside the renamed pin.
#[test]
fn a_claim_behind_the_rename_of_the_keys_own_pin_answers_the_renamed_pin() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
        // The decision read the pin, found no row for it and chose 2.
        let seen = table.lookup(s(1), 20, &rule);
        assert_eq!(seen, [WorkerId(5), WorkerId(0)]);
        table.rename_worker(WorkerId(5), WorkerId(15));
        let renamed = [WorkerId(15), WorkerId(0)];
        let claim = table.claim(s(1), seen, WorkerId(2));
        assert_eq!(claim, Claim::Standing(WorkerId(15), renamed));
        assert_eq!(table.pins(s(1)), Some(renamed), "nothing is claimed");
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
        let moved = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(15)],
            Recovery::Original => [WorkerId(15), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(moved));
    }
}

/// A rename says the id that stands ahead of its walk of the shards,
/// so a pin the walk has yet to reach holds an id its worker had. A
/// claim of that worker takes the walk's step for the key first: the
/// pin is read by the id that stands, the claim finds the worker
/// pinned and answers the decision's own pick, and no pin is written
/// beside it. In both orders: a worker registered twice and claimed
/// under its first id while the second rename is half walked, and a
/// worker registered once and claimed under its new id while that
/// rename is half walked. Written beside the pin, the claim would be
/// made one with it by the walk, and the claiming decision's commit
/// on an alternate would withdraw a pin the key was served on.
#[test]
fn a_claim_beside_a_pin_the_renames_walk_has_yet_to_reach_takes_the_walks_step() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        // Key 1 is served on a worker registered once since, key 2 on
        // a worker under its first id.
        for (key, worker) in [(1, 1), (2, 3)] {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        // Each worker's next rename by its first half: remembered,
        // the shards not yet walked.
        table.remember(WorkerId(11), WorkerId(21));
        table.remember(WorkerId(3), WorkerId(13));
        // Key 1's decision read the worker's first id, key 2's the
        // rows that stand: (key, the pin as held, the id read, the id
        // that stands).
        let orders = [(1, 11, 1, 21), (2, 3, 13, 13)];
        for (key, held, read, stands) in orders {
            let pins = [WorkerId(held), WorkerId(0)];
            assert_eq!(table.lookup(s(key), 20, &rule), pins);
            let claim = table.claim(s(key), pins, WorkerId(read));
            let led = [WorkerId(stands), WorkerId(0)];
            assert_eq!(claim, Claim::Standing(WorkerId(read), led), "key {key}");
            assert_eq!(table.pins(s(key)), Some(led), "key {key}: the step alone");
        }
        table.rename_worker(WorkerId(11), WorkerId(21));
        table.rename_worker(WorkerId(3), WorkerId(13));
        for (key, _, read, stands) in orders {
            let led = [WorkerId(stands), WorkerId(0)];
            assert_eq!(table.pins(s(key)), Some(led), "key {key} behind the walk");
            // The decision's dispatch on its pick is one on the pin.
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(read));
            assert_eq!(table.pins(s(key)), Some(led), "key {key} on its pick");
            // Its commit on an alternate withdraws nothing.
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(2));
            let both = match rule.recovery {
                Recovery::Latest => [WorkerId(2), WorkerId(stands)],
                Recovery::Original => [WorkerId(stands), WorkerId(2)],
            };
            assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
        }
    }
}

/// A pair's moved claim beside such a pin takes the walk's step too:
/// the prefill its pair formed on is the key's pin, under an id the
/// second rename's walk has yet to reach, and the decision read the
/// first. The pin takes the id that stands, and nothing is claimed.
#[test]
fn a_moved_claim_beside_a_pin_the_renames_walk_has_yet_to_reach_takes_the_walks_step() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        table.rename_worker(WorkerId(1), WorkerId(11));
        let pins = [WorkerId(11), WorkerId(0)];
        // The decision claims another prefill, then pairs on the
        // worker under its first id, the second rename half walked.
        let claim = table.claim(s(1), pins, WorkerId(5));
        assert_eq!(claim, Claim::Made(WorkerId(5)));
        table.remember(WorkerId(11), WorkerId(21));
        let verdict = table.move_claim(s(1), WorkerId(5), WorkerId(1));
        assert_eq!(verdict, SessionVerdict::Pinned);
        let led = [WorkerId(21), WorkerId(0)];
        assert_eq!(table.pins(s(1)), Some(led), "the pin alone, no claim");
        table.rename_worker(WorkerId(11), WorkerId(21));
        // The pair's commit on an alternate withdraws nothing.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
        let both = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(21)],
            Recovery::Original => [WorkerId(21), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(both));
    }
}

/// A commit and a respill's move on an id a worker had meet the
/// worker's own pin where a rename's walk has yet to reach it: the
/// pin takes the id that stands, the worker is pinned once, and the
/// key's other pin stays.
#[test]
fn a_commit_and_a_move_beside_a_pin_the_walk_has_yet_to_reach_pin_the_worker_once() {
    let table = SessionTable::new(RULE.sweep_micros);
    for key in [1, 2] {
        table.lookup(s(key), 10, &RULE);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(1));
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(4));
    }
    table.rename_worker(WorkerId(1), WorkerId(11));
    table.remember(WorkerId(11), WorkerId(21));
    assert_eq!(table.pins(s(1)), Some([WorkerId(4), WorkerId(11)]));
    // A commit on the worker's first id is the failover taken.
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    let taken = Some([WorkerId(4), WorkerId(21)]);
    assert_eq!(table.pins(s(1)), taken, "the pins keep their order");
    // A respill's move there fronts the worker's own pin.
    table.respilled(s(2), WorkerId(1));
    let fronted = Some([WorkerId(21), WorkerId(4)]);
    assert_eq!(table.pins(s(2)), fronted);
    table.rename_worker(WorkerId(11), WorkerId(21));
    assert_eq!(table.pins(s(1)), taken);
    assert_eq!(table.pins(s(2)), fronted);
    // A commit on a worker's new id inside its first rename's walk:
    // the key's leading pin is the worker's, and the failover stays.
    table.lookup(s(3), 10, &RULE);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(4));
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(5));
    table.remember(WorkerId(5), WorkerId(15));
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(15));
    let pins = Some([WorkerId(15), WorkerId(4)]);
    assert_eq!(table.pins(s(3)), pins, "the failover stays");
    table.rename_worker(WorkerId(5), WorkerId(15));
    assert_eq!(table.pins(s(3)), pins);
}

/// A key that holds one worker under two ids, half walked: its pin
/// under the id the worker had, and a claim written under the new id
/// ahead of the table's word of the rename. Inside that rename's walk
/// a dispatch on an id the worker had is one on the claimed worker,
/// and a respill fronts another worker. The first write that asks
/// takes the walk's step for the key: the two pins are one, and the
/// claim's mark is gone. The claiming decision's commit on the
/// respill's worker then withdraws nothing, with the dispatch between
/// the two and without it, as behind the whole walk.
#[test]
fn a_key_that_holds_one_worker_twice_is_walked_by_its_first_write_that_asks() {
    for rule in [RULE, ORIGINAL] {
        for dispatch in [true, false] {
            let said = (rule.recovery, dispatch);
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
            table.rename_worker(WorkerId(1), WorkerId(11));
            let served = [WorkerId(11), WorkerId(0)];
            // The decision read the row that replaced the worker's,
            // ahead of the table's word of that rename.
            let claim = table.claim(s(1), served, WorkerId(21));
            assert_eq!(claim, Claim::Made(WorkerId(21)), "{said:?}");
            // The rename's first half: remembered, the shards not
            // yet walked.
            table.remember(WorkerId(11), WorkerId(21));
            if dispatch {
                table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
                let one = [WorkerId(21), WorkerId(0)];
                assert_eq!(table.pins(s(1)), Some(one), "{said:?}: one pin");
            }
            table.respilled(s(1), WorkerId(2));
            let moved = [WorkerId(2), WorkerId(21)];
            assert_eq!(table.pins(s(1)), Some(moved), "{said:?}: the move");
            table.rename_worker(WorkerId(11), WorkerId(21));
            let claimed = SessionVerdict::Claimed(WorkerId(21));
            table.dispatched(s(1), claimed, WorkerId(2));
            assert_eq!(table.pins(s(1)), Some(moved), "{said:?}: the pin stands");
        }
    }
}

/// A claim's mark outlives its pin where a respill drops the pin: the
/// mark then names an id no pin of the key holds. Under `original` a
/// claim written ahead of the table's word of a rename stands behind
/// the key's served pin, and a respill ahead of that word drops it.
/// The rename brings the served pin under the mark's id, and the mark
/// goes there: that pin is no claim, and the claiming decision's
/// commit on the respill's worker withdraws nothing.
#[test]
fn a_mark_whose_pin_a_respill_dropped_goes_where_a_rename_brings_a_pin_under_it() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    let served = [WorkerId(1), WorkerId(0)];
    let claim = table.claim(s(1), served, WorkerId(11));
    assert_eq!(claim, Claim::Made(WorkerId(11)));
    table.respilled(s(1), WorkerId(2));
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(1)]));
    table.rename_worker(WorkerId(1), WorkerId(11));
    let moved = [WorkerId(2), WorkerId(11)];
    assert_eq!(table.pins(s(1)), Some(moved));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(2));
    assert_eq!(table.pins(s(1)), Some(moved), "the served pin stands");
}

/// The limit of a claim written ahead of the table's word of a
/// rename, held as it stands. Under `latest` the claim leads the
/// key's served pin, and a respill ahead of that word drops the
/// served pin as it drops any failover: nothing in the table says the
/// two ids are one worker yet. The claim stands unconfirmed behind
/// the rename, and the claiming decision's commit on the respill's
/// worker withdraws it: the key holds no pin on that worker, where a
/// respill behind the walk keeps one.
#[test]
fn a_respill_ahead_of_the_tables_word_drops_the_served_pin_under_latest() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    let served = [WorkerId(1), WorkerId(0)];
    let claim = table.claim(s(1), served, WorkerId(11));
    assert_eq!(claim, Claim::Made(WorkerId(11)));
    table.respilled(s(1), WorkerId(2));
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(11)]));
    table.rename_worker(WorkerId(1), WorkerId(11));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(2));
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(0)]));
}

/// A pair's moved claim reads its claim on the walked entry: the key
/// holds its served pin under the id a prefill had and the decision's
/// claim under the new id, written ahead of the table's word of the
/// rename, and the pair forms on another prefill, or on none, inside
/// the rename's walk. The key takes the walk's step ahead of the
/// move's own lines: the two pins are one and the mark is gone, so
/// the claim is a pin the key was served on, nothing moves and
/// nothing is claimed, as behind the whole walk. The pair's dispatch
/// moves the pins as a pin's alternate does.
#[test]
fn a_moved_claim_of_a_key_that_holds_one_worker_twice_is_read_on_the_walked_entry() {
    for rule in [RULE, ORIGINAL] {
        // The pair forms on another prefill, then on none.
        for paired in [5, 0] {
            let said = (rule.recovery, paired);
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
            table.rename_worker(WorkerId(1), WorkerId(11));
            let served = [WorkerId(11), WorkerId(0)];
            let claim = table.claim(s(1), served, WorkerId(21));
            assert_eq!(claim, Claim::Made(WorkerId(21)), "{said:?}");
            table.remember(WorkerId(11), WorkerId(21));
            let verdict = table.move_claim(s(1), WorkerId(21), WorkerId(paired));
            assert_eq!(verdict, SessionVerdict::Pinned, "{said:?}");
            let one = [WorkerId(21), WorkerId(0)];
            assert_eq!(table.pins(s(1)), Some(one), "{said:?}: one pin");
            table.rename_worker(WorkerId(11), WorkerId(21));
            assert_eq!(table.pins(s(1)), Some(one), "{said:?}");
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
            let both = match rule.recovery {
                Recovery::Latest => [WorkerId(5), WorkerId(21)],
                Recovery::Original => [WorkerId(21), WorkerId(5)],
            };
            assert_eq!(table.pins(s(1)), Some(both), "{said:?}");
        }
    }
}

/// A dispatch on a claim's own lease asks by the claim of record.
/// Under `original` a respill ahead of the table's word of a rename
/// drops the claim's pin, and the key keeps its served pin under the
/// id the worker had. Inside the rename's walk the claiming
/// decision's dispatch takes the walk's step for the key: the served
/// pin is the claimed worker's, the dispatch is the failover taken,
/// and the respill's worker stays, as behind the whole walk. Asked by
/// nothing, the dispatch would write the worker again and drop that
/// pin.
#[test]
fn a_dispatch_on_a_claims_lease_meets_the_keys_pin_of_its_worker_inside_the_walk() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    let served = [WorkerId(1), WorkerId(0)];
    let claim = table.claim(s(1), served, WorkerId(11));
    assert_eq!(claim, Claim::Made(WorkerId(11)));
    table.respilled(s(1), WorkerId(2));
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(1)]));
    // The rename's first half: remembered, the shards not yet walked.
    table.remember(WorkerId(1), WorkerId(11));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(11));
    let taken = [WorkerId(2), WorkerId(11)];
    assert_eq!(table.pins(s(1)), Some(taken), "the failover taken");
    table.rename_worker(WorkerId(1), WorkerId(11));
    assert_eq!(table.pins(s(1)), Some(taken));
    // The claim is confirmed: a later commit elsewhere withdraws none.
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(2));
    assert_eq!(table.pins(s(1)), Some(taken));
}

/// A claim reads the key's pins once more behind a rename's step. The
/// decision read the pin under the id its worker had, the rename's
/// walk has yet to reach the key, and the decision's pick is another
/// worker. The claim's step renames the pin, so the pins are not the
/// ones the decision read: the renamed pin answers as their newcomer,
/// and nothing is claimed, as where the whole walk came between the
/// read and the claim.
#[test]
fn a_claim_whose_step_renames_a_pin_answers_the_renamed_pin() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        // The rename's first half: remembered, the shards not yet
        // walked.
        table.remember(WorkerId(1), WorkerId(11));
        let read = [WorkerId(1), WorkerId(0)];
        assert_eq!(table.lookup(s(1), 20, &rule), read);
        let claim = table.claim(s(1), read, WorkerId(2));
        let led = [WorkerId(11), WorkerId(0)];
        assert_eq!(claim, Claim::Standing(WorkerId(11), led));
        assert_eq!(table.pins(s(1)), Some(led), "nothing is claimed");
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_eq!(table.pins(s(1)), Some(led));
    }
}

/// Where the claim's step makes two pins of one worker one, the pins
/// are not the ones the decision read and hold no newcomer: the
/// answer is the decision's own pick, and nothing is claimed, as
/// where the whole walk came between the read and the claim.
#[test]
fn a_claim_whose_step_makes_two_pins_one_answers_its_own_pick() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        // Ahead of the table's word: a claim on the row that
        // replaced the worker's.
        let served = [WorkerId(1), WorkerId(0)];
        let claim = table.claim(s(1), served, WorkerId(11));
        assert_eq!(claim, Claim::Made(WorkerId(11)));
        let read = table.lookup(s(1), 20, &rule);
        assert!(read.contains(&WorkerId(1)) && read.contains(&WorkerId(11)));
        table.remember(WorkerId(1), WorkerId(11));
        let claim = table.claim(s(1), read, WorkerId(2));
        let one = [WorkerId(11), WorkerId(0)];
        assert_eq!(claim, Claim::Standing(WorkerId(2), one));
        assert_eq!(table.pins(s(1)), Some(one), "nothing is claimed");
    }
}

/// Pins that moved under a decision are read behind a pending
/// rename's step. A host replaces a worker's row before it tells the
/// table. A decision reads the key's pin on the new rows, finds it
/// not routable and picks worker 2. A request decided on the earlier
/// rows commits on the replaced worker, under the id it had. The
/// rename is then said, and its walk has yet to reach the key. The
/// decision's claim names the newcomer by the id that stands, the
/// one its rows hold, and nothing is claimed. So too for key 2,
/// where another request committed on the decision's own pick as
/// well: a pin as written, which a write asks nothing for.
#[test]
fn a_claim_on_pins_that_moved_names_the_newcomer_by_the_id_that_stands() {
    for rule in [RULE, ORIGINAL] {
        let latest = rule.recovery == Recovery::Latest;
        let table = SessionTable::new(rule.sweep_micros);
        // (key, the commits between the decision's read and its
        // claim, the pins behind the step). Key 2's commits come in
        // the order that leaves the replaced worker in the lead.
        let keys: [(u64, &[u64], [u64; 2]); 2] = [
            (1, &[1], if latest { [11, 5] } else { [5, 11] }),
            (2, if latest { &[2, 1] } else { &[1, 2] }, [11, 2]),
        ];
        let read = [WorkerId(5), WorkerId(0)];
        for (key, commits, _) in keys {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(5));
            assert_eq!(table.lookup(s(key), 20, &rule), read);
            for worker in commits {
                table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(*worker));
            }
        }
        // The rename's first half: remembered, the shards not yet
        // walked.
        table.remember(WorkerId(1), WorkerId(11));
        for (key, _, walked) in keys {
            let walked = walked.map(WorkerId);
            assert_ne!(table.pins(s(key)), Some(walked), "key {key} is not walked");
            let claim = table.claim(s(key), read, WorkerId(2));
            assert_eq!(claim, Claim::Standing(WorkerId(11), walked), "key {key}");
            assert_eq!(
                table.pins(s(key)),
                Some(walked),
                "key {key}: the step alone"
            );
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        for (key, _, walked) in keys {
            let walked = walked.map(WorkerId);
            assert_eq!(
                table.pins(s(key)),
                Some(walked),
                "key {key} behind the walk"
            );
        }
    }
}

/// Pins that moved with no rename said are read as written, and
/// their first newcomer is the pin to follow, where the decision's
/// own pick is the second: two other requests committed, the later
/// one on the worker the decision picked. A worker that is a pin
/// already answers the decision's own pick only where the pins stand
/// as the decision read them.
#[test]
fn a_claim_on_pins_that_moved_follows_their_first_newcomer_past_its_own_pick() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
        let read = [WorkerId(5), WorkerId(0)];
        assert_eq!(table.lookup(s(1), 20, &rule), read);
        // The two commits, in the order that leaves worker 9 in the
        // lead and the decision's pick behind it.
        let commits = match rule.recovery {
            Recovery::Latest => [7, 9],
            Recovery::Original => [9, 7],
        };
        for worker in commits {
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(worker));
        }
        let held = [WorkerId(9), WorkerId(7)];
        assert_eq!(table.pins(s(1)), Some(held));
        let claim = table.claim(s(1), read, WorkerId(7));
        assert_eq!(claim, Claim::Standing(WorkerId(9), held));
        assert_eq!(table.pins(s(1)), Some(held), "nothing is written");
    }
}

/// Pins the step makes the ones the decision read again stand as
/// read. One worker is registered three times around one decision:
/// it was 3 where the decision read its rows and 1 where the
/// decision read the key's pins, a rename to 11 walked the key, and
/// a rename back to 1 is said and has yet to walk it. The pins as
/// written are not the ones the decision read, and the step makes
/// them so. The decision's pick is that worker under its first id, a
/// pin of the key already: the table writes nothing and answers the
/// decision's own pick, as where the pins stood as written. A claim
/// there would mark a pin the key was served on, and the decision's
/// commit on an alternate would withdraw it.
#[test]
fn a_claim_on_pins_the_step_makes_the_ones_read_again_meets_its_workers_pin() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.rename_worker(WorkerId(3), WorkerId(1));
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let read = table.lookup(s(1), 20, &rule);
        let served = match rule.recovery {
            Recovery::Latest => [WorkerId(1), WorkerId(5)],
            Recovery::Original => [WorkerId(5), WorkerId(1)],
        };
        assert_eq!(read, served);
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_ne!(table.pins(s(1)), Some(read), "the walk renamed the pin");
        // The rename back by its first half: remembered, the shards
        // not yet walked.
        table.remember(WorkerId(11), WorkerId(1));
        let claim = table.claim(s(1), read, WorkerId(3));
        assert_eq!(claim, Claim::Standing(WorkerId(3), read));
        assert_eq!(table.pins(s(1)), Some(read), "the step alone");
        // The request's commit on an alternate withdraws nothing.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
        let both = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(1)],
            Recovery::Original => [WorkerId(1), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(both), "the key keeps its pin");
    }
}

/// Pins the step makes the ones the decision read again stand as
/// read for a pick that is in neither pin too: the claim is taken, as
/// where the pins stood as written. One worker is registered twice
/// around one decision: it was 1 where the decision read the key's
/// pins, a rename to 11 walked the key, and a rename back to 1 is
/// said and has yet to walk it. The step makes the pins the ones
/// read, and the decision's pick, another worker, is claimed by the
/// key's recovery. The claim is marked: the claiming decision's
/// commit elsewhere withdraws it, and the key's pin stays.
#[test]
fn a_claim_on_pins_the_step_makes_the_ones_read_again_is_taken_for_a_new_worker() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let read = table.lookup(s(1), 20, &rule);
        assert!(read.contains(&WorkerId(1)) && read.contains(&WorkerId(5)));
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_ne!(table.pins(s(1)), Some(read), "the walk renamed the pin");
        // The rename back by its first half: remembered, the shards
        // not yet walked.
        table.remember(WorkerId(11), WorkerId(1));
        let claim = table.claim(s(1), read, WorkerId(2));
        assert_eq!(claim, Claim::Made(WorkerId(2)));
        let claimed = match rule.recovery {
            Recovery::Latest => [WorkerId(2), WorkerId(1)],
            Recovery::Original => [WorkerId(1), WorkerId(2)],
        };
        assert_eq!(table.pins(s(1)), Some(claimed), "on the walked pins");
        // The claiming decision's commit lands elsewhere.
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(7));
        let withdrawn = match rule.recovery {
            Recovery::Latest => [WorkerId(7), WorkerId(1)],
            Recovery::Original => [WorkerId(1), WorkerId(7)],
        };
        assert_eq!(table.pins(s(1)), Some(withdrawn), "the claim is withdrawn");
    }
}

/// A decision that claims a worker it read as a pin of the key, on
/// pins that stand as it read them, makes no claim: the table writes
/// nothing, marks nothing and answers the decision's own pick, as for
/// a worker pinned under another id. The decision's commit on that
/// pick then leaves the pins as read, as a failover taken does, and
/// its commit elsewhere moves them as a pin's alternate does. The
/// keyed decision answers a pin it read ahead of its claim, and this
/// is the table's answer to a caller that does not. The gateway's
/// sticky map writes nothing where a key's request lands on a
/// candidate it holds
/// (`model_gateway/src/policies/registry.rs:794-797`).
#[test]
fn a_claim_of_a_pin_its_decision_read_is_no_claim() {
    for rule in [RULE, ORIGINAL] {
        // The key's table up to the claim, made once for each commit.
        let claimed = || {
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
            let read = table.lookup(s(1), 20, &rule);
            // The decision picks the pin it read behind the lead.
            let pick = read[1];
            let claim = table.claim(s(1), read, pick);
            assert_eq!(claim, Claim::Standing(pick, read));
            assert_eq!(table.pins(s(1)), Some(read), "nothing is written");
            (table, read, pick)
        };
        // The decision's commit lands on its pick: the failover taken.
        let (table, read, pick) = claimed();
        table.dispatched(s(1), SessionVerdict::Pinned, pick);
        assert_eq!(table.pins(s(1)), Some(read), "the pins stay as read");
        // The decision's commit lands elsewhere, under a pin's verdict.
        let (table, read, pick) = claimed();
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(9));
        let moved = match rule.recovery {
            Recovery::Latest => [WorkerId(9), read[0]],
            Recovery::Original => [pick, WorkerId(9)],
        };
        assert_eq!(table.pins(s(1)), Some(moved), "as a pin's alternate");
    }
}

/// A key that holds one pin keeps it where the decision that read it
/// claims it and is sent elsewhere. The table marks no claim, so the
/// commit withdraws nothing, even one that names a claim: the worker
/// the key was served on stands behind the newcomer by the key's
/// recovery. A marked pin would go with that commit and the key would
/// forget its worker. The gateway's sticky map pins a new worker first
/// and keeps the one before it as its second candidate
/// (`model_gateway/src/policies/manual.rs:269-274`), and under its own
/// assignments the new worker joins behind (`manual.rs:125-130`).
#[test]
fn a_claim_of_a_keys_one_pin_leaves_it_behind_a_commit_elsewhere() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
        let read = table.lookup(s(1), 20, &rule);
        assert_eq!(read, [WorkerId(5), WorkerId(0)]);
        let claim = table.claim(s(1), read, WorkerId(5));
        assert_eq!(claim, Claim::Standing(WorkerId(5), read));
        assert_eq!(table.pins(s(1)), Some(read), "nothing is written");
        // A commit elsewhere that names a claim finds no mark.
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(5)), WorkerId(9));
        let kept = match rule.recovery {
            Recovery::Latest => [WorkerId(9), WorkerId(5)],
            Recovery::Original => [WorkerId(5), WorkerId(9)],
        };
        assert_eq!(table.pins(s(1)), Some(kept), "the served pin stays");
    }
}

/// A pin the decision read under an id its worker had is no claim
/// either, ahead of the rename's walk. The key was served on worker 1,
/// the decision read that pin and its rows, and a rename of 1 to 11
/// is said with its walk pending. The decision's pick is 1, a pin as
/// written: the table takes no step, writes nothing and answers the
/// decision's own pick. The walk then renames the pin, and a commit
/// elsewhere that names a claim leaves the worker's pin under the id
/// that stands. A marked pin would follow the rename and go with that
/// commit. The gateway's sticky map names a key's workers by address
/// (`model_gateway/src/policies/manual.rs:120`, read by `:471-473`),
/// so a worker registered again is the key's candidate still.
#[test]
fn a_claim_of_a_pin_read_under_an_id_its_worker_had_is_no_claim() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let read = table.lookup(s(1), 20, &rule);
        assert_eq!(read, [WorkerId(1), WorkerId(0)]);
        // The rename by its first half: remembered, the shards not
        // yet walked.
        table.remember(WorkerId(1), WorkerId(11));
        let claim = table.claim(s(1), read, WorkerId(1));
        assert_eq!(claim, Claim::Standing(WorkerId(1), read));
        assert_eq!(table.pins(s(1)), Some(read), "no step and no write");
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
        // A commit elsewhere that names a claim finds no mark.
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(9));
        let kept = match rule.recovery {
            Recovery::Latest => [WorkerId(9), WorkerId(11)],
            Recovery::Original => [WorkerId(11), WorkerId(9)],
        };
        assert_eq!(table.pins(s(1)), Some(kept), "the worker's pin stays");
    }
}

/// A pin the decision read is no claim on pins a rename's step makes
/// the ones read again either. One worker is registered twice around
/// one decision: it was 1 where the decision read the key's pins, a
/// rename to 11 walked the key, and a rename back to 1 is said and
/// has yet to walk it. The decision's pick is that worker, a pin it
/// read. The step makes the pins the ones read, and the table writes
/// nothing more and answers the decision's own pick. A commit
/// elsewhere that names a claim then keeps the worker's pin. A marked
/// pin would go with that commit. The gateway's sticky map names a
/// key's workers by address
/// (`model_gateway/src/policies/manual.rs:120`, read by `:471-473`),
/// so a worker registered again twice is the key's candidate still.
#[test]
fn a_claim_of_a_pin_its_decision_read_is_no_claim_behind_a_renames_step() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let read = table.lookup(s(1), 20, &rule);
        assert!(read.contains(&WorkerId(1)) && read.contains(&WorkerId(5)));
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_ne!(table.pins(s(1)), Some(read), "the walk renamed the pin");
        // The rename back by its first half: remembered, the shards
        // not yet walked.
        table.remember(WorkerId(11), WorkerId(1));
        let claim = table.claim(s(1), read, WorkerId(1));
        assert_eq!(claim, Claim::Standing(WorkerId(1), read));
        assert_eq!(table.pins(s(1)), Some(read), "the step alone");
        // A commit elsewhere that names a claim finds no mark.
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(9));
        let kept = match rule.recovery {
            Recovery::Latest => [WorkerId(9), WorkerId(1)],
            Recovery::Original => [WorkerId(1), WorkerId(9)],
        };
        assert_eq!(table.pins(s(1)), Some(kept), "the worker's pin stays");
    }
}

/// A claim written ahead of the table's word of a rename finds no
/// rename to ask: a host replaces a worker's row before it tells the
/// table, and a decision between the two reads a pin no pool holds
/// and is placed by the rule. The key then holds the worker twice, as
/// a claim and as a pin, in both orders: a claim on the new row beside
/// the pin under the id the worker had, and a commit on the new row
/// beside a claim under that id. The rename's walk makes the two pins
/// one and the claim's mark goes there: the key holds the worker as a
/// pin, and the claiming decision's commit on an alternate withdraws
/// nothing.
#[test]
fn a_claim_made_one_with_the_keys_pin_by_a_rename_is_confirmed() {
    for rule in [RULE, ORIGINAL] {
        let table = SessionTable::new(rule.sweep_micros);
        // Key 1 was served on the worker and is claimed on the row
        // that replaced it.
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let served = [WorkerId(1), WorkerId(0)];
        let claim = table.claim(s(1), served, WorkerId(11));
        assert_eq!(claim, Claim::Made(WorkerId(11)));
        // Key 2 is claimed on the worker and served on the row that
        // replaced it.
        table.lookup(s(2), 10, &rule);
        let claim = table.claim(s(2), [WorkerId(0); 2], WorkerId(1));
        assert_eq!(claim, Claim::Made(WorkerId(1)));
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(11));
        table.rename_worker(WorkerId(1), WorkerId(11));
        // (key, the id its claim's lease names)
        for (key, leased) in [(1, 11), (2, 1)] {
            let led = [WorkerId(11), WorkerId(0)];
            assert_eq!(table.pins(s(key)), Some(led), "key {key} holds one pin");
            // The claiming decision's commit on an alternate.
            let claimed = SessionVerdict::Claimed(WorkerId(leased));
            table.dispatched(s(key), claimed, WorkerId(2));
            let both = match rule.recovery {
                Recovery::Latest => [WorkerId(2), WorkerId(11)],
                Recovery::Original => [WorkerId(11), WorkerId(2)],
            };
            assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
        }
    }
}

/// A claim's commit elsewhere withdraws it inside a rename's walk too:
/// the renames already say the new id, the entry still holds the old
/// one, and the claim is matched by the id its lease names.
#[test]
fn a_claims_commit_elsewhere_inside_the_renames_walk_withdraws_it() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(9), 10, &ORIGINAL);
    let claim = table.claim(s(9), [WorkerId(0); 2], WorkerId(1));
    assert_eq!(claim, Claim::Made(WorkerId(1)));
    // The rename's first half: remembered, the shard not yet walked.
    table.remember(WorkerId(1), WorkerId(11));
    table.dispatched(s(9), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
    table.rename_worker(WorkerId(1), WorkerId(11));
    assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(0)]));
}

/// A claim outlives the renames the table forgets: the id its lease
/// names is on its entry, so its commit elsewhere withdraws it behind
/// a list that started over.
#[test]
fn a_claim_renamed_ahead_of_the_bound_is_withdrawn_by_its_commit() {
    let table = SessionTable::new(ORIGINAL.sweep_micros);
    table.lookup(s(9), 10, &ORIGINAL);
    let claim = table.claim(s(9), [WorkerId(0); 2], WorkerId(1));
    assert_eq!(claim, Claim::Made(WorkerId(1)));
    table.rename_worker(WorkerId(1), WorkerId(11));
    let bound = RENAMED_WORKERS as u64;
    for life in 0..bound {
        table.rename_worker(WorkerId(100_000 + life), WorkerId(200_000 + life));
    }
    assert_eq!(table.renamed(), 1, "the list started over");
    table.dispatched(s(9), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
    assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(0)]));
}

/// The table remembers a bounded count of renames: at the bound the
/// list starts over, and the rename that follows is remembered.
#[test]
fn the_renames_the_table_remembers_are_bounded() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    let bound = RENAMED_WORKERS as u64;
    for life in 1..=bound {
        table.rename_worker(WorkerId(life), WorkerId(life + 1));
    }
    assert_eq!(table.renamed(), RENAMED_WORKERS);
    table.rename_worker(WorkerId(bound + 1), WorkerId(bound + 2));
    assert_eq!(table.renamed(), 1);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(bound + 1));
    assert_eq!(table.pins(s(1)), Some([WorkerId(bound + 2), WorkerId(0)]));
}

/// A worker that left keeps its pins. Its key is served elsewhere
/// meanwhile, and the worker registered again under a new id leads the
/// key again under `original`.
#[test]
fn a_worker_that_left_keeps_its_pins_and_takes_them_back_under_its_new_id() {
    let table = SessionTable::new(RULE.sweep_micros);
    for (key, worker) in [(1, 1), (2, 2)] {
        table.lookup(s(key), 10, &ORIGINAL);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
    }
    table.left(WorkerId(1));
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    table.lookup(s(1), 20, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
    table.rename_worker(WorkerId(1), WorkerId(11));
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(2)]));
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
}

/// Under `latest` the worker that served the key in the absence keeps
/// the lead, and the one registered again is its failover.
#[test]
fn under_latest_recovery_a_worker_registered_again_is_the_failover() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 10, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    table.left(WorkerId(1));
    table.lookup(s(1), 20, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
    table.rename_worker(WorkerId(1), WorkerId(11));
    assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(11)]));
}

/// A worker that stays away is forgotten once the absence's time
/// passes on the decision clock, counted from the first decision
/// behind the leave. A table no rule sweeps counts it too.
#[test]
fn a_worker_that_stays_away_is_forgotten_at_the_absences_end() {
    for sweep in [RULE.sweep_micros, 0] {
        let rule = SessionRule {
            ttl_micros: None,
            sweep_micros: sweep,
            ..ORIGINAL
        };
        let table = SessionTable::new(sweep).with_absence(500);
        for (key, worker) in [(1, 1), (2, 2)] {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
        }
        table.left(WorkerId(1));
        // The decision at 100 stamps the end: 600.
        table.lookup(s(2), 100, &rule);
        table.lookup(s(2), 599, &rule);
        let stands = Some([WorkerId(1), WorkerId(0)]);
        assert_eq!(table.pins(s(1)), stands, "inside its time, sweep {sweep}");
        table.lookup(s(2), 600, &rule);
        assert_eq!(table.pins(s(1)), None, "its only pin went, sweep {sweep}");
        assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
    }
}

/// A worker registered again inside its time is not forgotten at its
/// end, under a new id or under the one it had.
#[test]
fn a_worker_registered_again_in_time_is_not_forgotten_at_the_absences_end() {
    let rule = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..ORIGINAL
    };
    let table = SessionTable::new(0).with_absence(500);
    table.lookup(s(1), 10, &rule);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
    table.left(WorkerId(1));
    table.lookup(s(1), 100, &rule);
    table.rename_worker(WorkerId(1), WorkerId(11));
    table.lookup(s(1), 10_000, &rule);
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
    table.left(WorkerId(11));
    table.rename_worker(WorkerId(11), WorkerId(11));
    table.lookup(s(1), 20_000, &rule);
    assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
}

/// A join under the id that left ends its absence, by its own word or
/// by a snapshot that lists it: no end of the absence takes the pins of
/// the worker that is back.
#[test]
fn a_join_under_the_id_that_left_ends_its_absence() {
    let rule = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..ORIGINAL
    };
    let table = SessionTable::new(0).with_absence(500);
    for (key, worker) in [(1, 1), (2, 2)] {
        table.lookup(s(key), 10, &rule);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
    }
    table.left(WorkerId(1));
    table.returned(WorkerId(1));
    assert_eq!(table.absent(), 0);
    table.lookup(s(2), 100, &rule);
    table.lookup(s(2), 10_000, &rule);
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    table.left(WorkerId(1));
    table.left(WorkerId(2));
    table.returned_all(&[WorkerId(2)]);
    assert_eq!(table.absent(), 1, "the one the snapshot does not list");
    table.lookup(s(2), 20_000, &rule);
    table.lookup(s(2), 30_000, &rule);
    assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
    assert_eq!(table.pins(s(1)), None, "the worker that stayed away");
}

/// A leave asks for a sweep, and that sweep walks no idle key ahead of
/// the pass over them: the pass keeps its own instant. The key past its
/// end stays until the pass is due, as with no leave.
#[test]
fn a_leave_walks_no_idle_key_before_their_pass_is_due() {
    let rule = SessionRule {
        ttl_micros: Some(100),
        sweep_micros: 1_000,
        ..ORIGINAL
    };
    let table = SessionTable::new(rule.sweep_micros).with_absence(5_000);
    // The first lookup runs the pass: the next is due at 1,010.
    for (key, worker) in [(1, 1), (2, 2)] {
        table.lookup(s(key), 10, &rule);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
    }
    table.left(WorkerId(9));
    // Key 1 is past its end, the sweep at 1,000, and the leave's sweep
    // runs.
    table.lookup(s(2), 1_005, &rule);
    assert_eq!(table.absent(), 1, "the leave is stamped");
    assert!(table.pins(s(1)).is_some(), "no pass before 1,010");
    table.lookup(s(2), 1_010, &rule);
    assert_eq!(table.pins(s(1)), None, "the pass at its own instant");
}

/// A table that holds no key files no absence: no pin names the
/// worker, and no decision would end the entry.
#[test]
fn a_table_that_holds_no_key_files_no_absence() {
    let table = SessionTable::new(0).with_absence(500);
    table.left(WorkerId(1));
    assert_eq!(table.absent(), 0);
    table.lookup(s(1), 10, &ORIGINAL);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
    table.left(WorkerId(1));
    assert_eq!(table.absent(), 1);
}

/// The absent workers are bounded: at the bound the one that left
/// first is forgotten at once, its pins with it.
#[test]
fn at_the_bound_of_absent_workers_the_first_to_leave_is_forgotten() {
    let rule = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..ORIGINAL
    };
    let table = SessionTable::new(0).with_absence(500);
    for (key, worker) in [(1, 1), (2, 9_999)] {
        table.lookup(s(key), 10, &rule);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
    }
    let bound = ABSENT_WORKERS as u64;
    for worker in 1..=bound {
        table.left(WorkerId(worker));
    }
    assert_eq!(table.absent(), ABSENT_WORKERS);
    assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    table.left(WorkerId(bound + 1));
    assert_eq!(table.absent(), ABSENT_WORKERS);
    assert_eq!(table.pins(s(1)), None, "the first to leave");
    assert_eq!(table.pins(s(2)), Some([WorkerId(9_999), WorkerId(0)]));
}

/// A session of the sizing cases: a counter mixed over every bit, as a
/// session's value is.
fn mixed(count: u64) -> Session {
    s(count.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
}

#[test]
fn a_table_of_any_size_keeps_every_key_and_is_sized_by_a_power_of_two() {
    for (asked, shards) in [(0, 1), (1, 1), (3, 4), (16, 16), (1_000, 1_024)] {
        let table = SessionTable::new(RULE.sweep_micros).with_shards(asked);
        assert_eq!(table.shards(), shards, "{asked} shards asked");
        for count in 1..=200 {
            let key = mixed(count);
            assert_eq!(table.lookup(key, 10, &RULE), [WorkerId(0); 2]);
            table.dispatched(key, SessionVerdict::Pinned, WorkerId(count));
            assert_eq!(table.lookup(key, 11, &RULE)[0], WorkerId(count));
        }
        assert_eq!(table.len(), 200);
        // A walk of the table reaches every shard.
        table.forget_workers(&[WorkerId(200)]);
        assert_eq!(table.pins(mixed(200)), None);
        assert_eq!(table.len(), 199);
    }
    // A table no host sized holds the count it always held.
    assert_eq!(SessionTable::new(0).shards(), SHARDS);
    assert!(shards_for_this_host().is_power_of_two());
    assert!(shards_for_this_host() >= 4);
}

/// The keys of a sized table spread over its shards: no shard holds
/// many times its share, so two decisions of different keys seldom
/// take one mutex.
#[test]
fn a_sized_tables_keys_spread_over_its_shards() {
    let table = SessionTable::new(0).with_shards(64);
    let rule = SessionRule {
        ttl_micros: None,
        sweep_micros: 0,
        ..RULE
    };
    let keys = 64 * 100;
    for count in 1..=keys {
        table.lookup(mixed(count), 10, &rule);
    }
    let held: Vec<usize> = table.shards.iter().map(|s| s.0.lock().len()).collect();
    assert_eq!(held.iter().sum::<usize>(), keys as usize);
    let (least, most) = (held.iter().min(), held.iter().max());
    assert!(
        least >= Some(&50) && most <= Some(&200),
        "a share of 100 a shard: {least:?} to {most:?}"
    );
}

/// A full table records no further key, in whichever shard it falls:
/// such a key reads no pin, is written none and claims none, while the
/// keys the table holds are answered. Keys that leave at their end give
/// their room back, and their shard its spare storage.
#[test]
fn a_full_table_records_no_further_key_and_answers_the_keys_it_holds() {
    let table = SessionTable::new(RULE.sweep_micros);
    for key in 1..=8 {
        table.lookup(s(key), 10, &RULE);
        table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(1));
    }
    // One key under the 4,194,304 a table holds: the count stands for
    // the keys no test inserts, and key 9 takes the last room.
    table.held.0.store(4_194_303, Ordering::Relaxed);
    table.lookup(s(9), 10, &RULE);
    table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1));
    assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS);
    // A key of another shard finds the table full.
    let fresh = s(1 << 32);
    let none = [WorkerId(0); 2];
    assert_eq!(table.lookup(fresh, 20, &RULE), none);
    assert_eq!(table.claim(fresh, none, WorkerId(2)), Claim::NoRoom);
    table.dispatched(fresh, SessionVerdict::Pinned, WorkerId(2));
    assert_eq!(table.pins(fresh), None);
    // A key the table holds is read and renewed as before.
    assert_eq!(table.lookup(s(1), 500, &RULE), [WorkerId(1), WorkerId(0)]);
    // The other eight end at 1 100 and leave at the pass.
    let room = table.shards[0].0.lock().capacity();
    table.sweep_due(1_200);
    assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS - 8);
    assert!(table.shards[0].0.lock().capacity() < room);
    assert_eq!(table.lookup(fresh, 1_200, &RULE), none);
    assert_eq!(
        table.claim(fresh, none, WorkerId(2)),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS - 7);
}

#[test]
fn a_shard_stands_on_cache_lines_of_its_own() {
    assert_eq!(align_of::<Shard>(), 128);
    assert_eq!(size_of::<Shard>(), 128);
}

/// A host that sweeps on a timer of its own drops the keys past their
/// end with no decision's read, and a sweep that is not due walks
/// nothing.
#[test]
fn a_hosts_sweep_drops_the_keys_past_their_end_with_no_read() {
    let table = SessionTable::new(RULE.sweep_micros).with_shards(8);
    table.lookup(mixed(1), 0, &RULE);
    table.dispatched(mixed(1), SessionVerdict::Pinned, WorkerId(1));
    table.lookup(mixed(2), 900, &RULE);
    table.dispatched(mixed(2), SessionVerdict::Pinned, WorkerId(2));
    // The read at 900 swept and set the next pass a sweep interval on:
    // a host's call ahead of it walks nothing.
    table.sweep_due(950);
    assert_eq!(table.len(), 2);
    // Past the interval and the first key's end.
    table.sweep_due(1_100);
    assert_eq!(table.pins(mixed(1)), None);
    assert_eq!(table.pins(mixed(2)), Some([WorkerId(2), WorkerId(0)]));
}

/// A read sweeps where a sweep is due until a pass of the table's host
/// has run. From the pass's first run a read walks no shard but its
/// key's, however long a sweep has been due: a key past its end reads
/// as no pins and is held until the pass, which drops it.
#[test]
fn a_read_leaves_every_sweep_to_the_hosts_pass_once_one_has_run() {
    let table = SessionTable::new(RULE.sweep_micros).with_shards(8);
    let pin = |key: u64, at: u64| {
        table.lookup(mixed(key), at, &RULE);
        table.dispatched(mixed(key), SessionVerdict::Pinned, WorkerId(key));
    };
    // No pass has run: the read at 1 100 sweeps the key past its end.
    pin(1, 0);
    pin(2, 1_100);
    assert_eq!(table.pins(mixed(1)), None, "a read swept");
    assert_eq!(table.len(), 1);
    // The pass runs, with no sweep due. Every sweep is its own now.
    table.sweep_due(1_150);
    assert_eq!(table.len(), 1);
    // Far past the second key's end and many sweep intervals on: reads
    // of other keys sweep nothing, and the key's own read finds none.
    pin(3, 9_000);
    pin(4, 9_500);
    assert_eq!(table.pins_at(mixed(2), 9_500), [WorkerId(0); 2]);
    assert_eq!(table.len(), 3, "the key past its end is held for the pass");
    table.sweep_due(9_600);
    assert_eq!(table.pins(mixed(2)), None, "the pass dropped it");
    assert_eq!(table.len(), 2);
}
