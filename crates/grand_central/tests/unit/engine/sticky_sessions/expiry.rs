//! A key's end and the sweeps: when a key's idle time ends, what a read and
//! a touch leave of that end, the sweeps that drop the keys past it, and the
//! two times a table takes from a document.

use super::*;

#[test]
fn idle_keys_are_swept_once_the_interval_passes() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(1), 0, &RULE);
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    table.lookup(s(2), 900, &RULE);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2), 0);
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
    forever.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
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
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
    table.lookup(s(2), 0, &RULE);
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2), 0);
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
        table.dispatched(mixed(key), SessionVerdict::Pinned, WorkerId(key), 0);
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
    table.dispatched(mixed(4), SessionVerdict::Pinned, WorkerId(1), 0);
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
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
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
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
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
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2), 0);
    assert_eq!(
        table.held(s(2)),
        Some(([WorkerId(2), WorkerId(0)], SessionExpiry::Never))
    );
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3), 0);
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
    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(7), 0);
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
    table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(9), 0);
    assert_eq!(
        table.pins_at(s(2), u64::MAX - 1),
        [WorkerId(9), WorkerId(0)]
    );
    assert_eq!(table.pins_at(s(2), u64::MAX), [WorkerId(0); 2]);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3), 0);
    assert_eq!(table.pins_at(s(3), 50), [WorkerId(0); 2]);
}

/// A key past its end is forgotten at its next lookup even when no
/// sweep ran since: the compare is the key's own, the walk only reclaims.
#[test]
fn an_expired_key_is_forgotten_at_its_lookup_before_any_sweep() {
    let rarely = SessionTable::new(1_000_000);
    rarely.lookup(s(1), 0, &RULE);
    rarely.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
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
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1), 0);
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
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1), 0);

    table.lookup(s(7), 1_000, &RULE);
    table.lookup(s(7), 600, &ORIGINAL);

    assert_eq!(
        table.held(s(7)),
        Some(([WorkerId(1), WorkerId(0)], SessionExpiry::At(2_000)))
    );
    assert_eq!(
        table.claim(s(7), [WorkerId(1), WorkerId(0)], WorkerId(2), 0),
        Claim::Made(WorkerId(2))
    );
    assert_eq!(table.lookup(s(7), 1_600, &RULE), [WorkerId(2), WorkerId(1)]);
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
    let claim = table.claim(s(1), [WorkerId(0); 2], WorkerId(3), 0);
    assert_eq!(claim, Claim::Made(WorkerId(3)));
    table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(3)), WorkerId(3), 0);
    assert_eq!(table.lookup(s(1), 1_020, &rule), [WorkerId(3), WorkerId(0)]);
}

/// A host that sweeps on a timer of its own drops the keys past their
/// end with no decision's read, and a sweep that is not due walks
/// nothing.
#[test]
fn a_hosts_sweep_drops_the_keys_past_their_end_with_no_read() {
    let table = SessionTable::new(RULE.sweep_micros).with_shards(8);
    table.lookup(mixed(1), 0, &RULE);
    table.dispatched(mixed(1), SessionVerdict::Pinned, WorkerId(1), 0);
    table.lookup(mixed(2), 900, &RULE);
    table.dispatched(mixed(2), SessionVerdict::Pinned, WorkerId(2), 0);
    // The read at 900 swept and set the next pass a sweep interval on:
    // a host's call ahead of it walks nothing.
    table.sweep_due(950);
    assert_eq!(table.len(), 2);
    // Past the interval and the first key's end.
    table.sweep_due(1_100);
    assert_eq!(table.pins(mixed(1)), None);
    assert_eq!(table.pins(mixed(2)), Some([WorkerId(2), WorkerId(0)]));
}
