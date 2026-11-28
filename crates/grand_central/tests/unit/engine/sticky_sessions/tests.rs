use super::*;

mod claims;
mod expiry;
mod membership;
mod shards;

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
    sides: SessionSides::Any,
};

const ORIGINAL: SessionRule = SessionRule {
    recovery: Recovery::Original,
    ..RULE
};

#[test]
fn the_latest_dispatch_leads_and_the_one_before_it_is_the_failover() {
    let table = SessionTable::new(RULE.sweep_micros);
    assert_eq!(table.lookup(s(7), 10, &RULE), [WorkerId(0); 2]);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1), 0);
    assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(0)]));
    // A hit on the preferred pin changes nothing.
    table.lookup(s(7), 20, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1), 0);
    assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(0)]));
    // A worker outside the pins (the pins were unroutable) leads.
    table.lookup(s(7), 30, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(2), 0);
    assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
    // The failover taken while the preferred one is unroutable keeps the
    // order; a respill to it makes it the preferred one.
    table.lookup(s(7), 40, &RULE);
    table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1), 0);
    assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
    table.lookup(s(7), 50, &RULE);
    table.dispatched(s(7), SessionVerdict::Respill, WorkerId(1), 0);
    assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(2)]));
}

#[test]
fn under_original_recovery_the_first_worker_keeps_the_lead() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.lookup(s(3), 10, &ORIGINAL);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(1), 0);
    table.lookup(s(3), 20, &ORIGINAL);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2), 0);
    assert_eq!(table.pins(s(3)), Some([WorkerId(1), WorkerId(2)]));
    // A third worker evicts the oldest.
    table.lookup(s(3), 30, &ORIGINAL);
    table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3), 0);
    assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(3)]));
}

#[test]
fn an_empty_key_or_target_writes_nothing() {
    let table = SessionTable::new(RULE.sweep_micros);
    table.dispatched(s(0), SessionVerdict::Pinned, WorkerId(1), 0);
    table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(0), 0);
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
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1), 0);
        table.respilled(s(7), WorkerId(2), 0);
        assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
        // The worker that leads already: nothing moves, and the commit
        // finds the pins in place.
        table.respilled(s(7), WorkerId(2), 0);
        table.dispatched(s(7), SessionVerdict::Respill, WorkerId(2), 0);
        assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
        table.respilled(s(0), WorkerId(3), 0);
        table.respilled(s(7), WorkerId(0), 0);
        table.respilled(s(8), WorkerId(3), 0);
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
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1), 0);
        // Both decisions read the pin ahead of either's move.
        let read = [20, 21].map(|now| table.lookup(s(7), now, &RULE));
        assert_eq!(read, [[WorkerId(1), WorkerId(0)]; 2]);
        if moved {
            table.respilled(s(7), WorkerId(2), 0);
            table.respilled(s(7), WorkerId(3), 0);
            assert_eq!(table.pins(s(7)), Some([WorkerId(3), WorkerId(2)]));
        }
        // The later decision's request is sent first.
        table.dispatched(s(7), SessionVerdict::Respill, WorkerId(3), 0);
        table.dispatched(s(7), SessionVerdict::Respill, WorkerId(2), 0);
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
                    assert_eq!(table.claim(s(1), seen, claimed, 0), Claim::Made(claimed));
                    if moved {
                        table.respilled(s(1), taken, 0);
                        assert_eq!(table.pins(s(1)), Some([taken, claimed]));
                    }
                    let claims = SessionVerdict::Claimed(claimed);
                    if claim_first {
                        table.dispatched(s(1), claims, lands, 0);
                    }
                    table.dispatched(s(1), SessionVerdict::Respill, taken, 0);
                    if !claim_first {
                        table.dispatched(s(1), claims, lands, 0);
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
    table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1), 0);
    table.lookup(s(9), 20, &RULE);
    table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(2), 0);
    assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(1)]));
    // A decides a respill onto the failover, B reads the key and takes
    // the failover as a plain hit, B commits first: B's hit leaves the
    // pins, A's respill fronts the failover whatever came between.
    table.lookup(s(9), 30, &RULE);
    table.lookup(s(9), 31, &RULE);
    table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1), 0);
    assert_eq!(
        table.pins(s(9)),
        Some([WorkerId(2), WorkerId(1)]),
        "B's dispatch onto the failover is a hit"
    );
    table.dispatched(s(9), SessionVerdict::Respill, WorkerId(1), 0);
    assert_eq!(
        table.pins(s(9)),
        Some([WorkerId(1), WorkerId(2)]),
        "A's dispatch is the respill"
    );
}

/// A session of the sizing cases: a counter mixed over every bit, as a
/// session's value is.
fn mixed(count: u64) -> Session {
    s(count.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
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
        table.dispatched(mixed(key), SessionVerdict::Pinned, WorkerId(key), 0);
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
