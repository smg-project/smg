use super::*;

/// A sessions rule that delegates runs the profile's own stages through
/// the profile run's one home: a keyed request's placement hands the
/// scorers the set the picker compares, as an unkeyed decision does.
#[test]
fn a_delegating_sessions_rule_narrows_as_the_profile_run_does() {
    let (engine, handed) = probed(Box::new(CountFirst), Some(SESSIONS));
    let pool = [
        candidate(0, 4),
        candidate(1, 2),
        candidate(2, 2),
        candidate(3, 9),
    ];
    let decision = decide_keyed(&engine, &pool, 41, 1_000);
    let pick = decision.pick().expect("a pick");
    assert_eq!(
        *handed.lock().expect("the probe's rows"),
        [1, 2],
        "the two that hold two"
    );
    let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
    assert_eq!(
        order,
        [2, 1, 0, 3],
        "the later of the fewest, then by count"
    );
    assert_eq!(pick.trace.survivors.get(), Some(4));
}

/// A key is placed by the profile once, then stays on its pin past the
/// load; an unroutable pin hands the key to a failover; once both are
/// routable the rule's recovery says which leads.
#[test]
fn a_keyed_request_stays_on_its_pin_and_recovers_by_the_rule() {
    for (recovery, back_to_first) in [(Recovery::Latest, false), (Recovery::Original, true)] {
        let engine = session_engine(SessionRule {
            recovery,
            ..SESSIONS
        });
        let mut pool = vec![candidate(0, 5), candidate(1, 0), candidate(2, 3)];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 1);
        pool[1].in_flight = 9;
        for t in 1..4 {
            assert_eq!(
                place_keyed(&engine, &pool, 7, 10 + t).position,
                1,
                "the pin holds past the load ({recovery:?})"
            );
        }
        pool[1].health.healthy = false;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 20).position,
            2,
            "the emptiest routable worker takes the key"
        );
        pool[1].health.healthy = true;
        let back = place_keyed(&engine, &pool, 7, 30);
        assert_eq!(back.position == 1, back_to_first, "{recovery:?}");
        assert_eq!(engine.sessions().len(), 1);
    }
}

/// Two pins, the owner leading and at the cap on the key. Where the
/// stages pick the owner again the request stays on it. Where they pick
/// the failover, under the cap, the respill goes there and the pin
/// follows it, so the failover leads once the owner is under the cap
/// again.
#[test]
fn a_saturated_owner_respills_onto_its_failover_and_the_failover_leads_after() {
    let engine = session_engine(SessionRule {
        recovery: Recovery::Original,
        ..SESSIONS
    });
    let mut pool = vec![candidate(0, 9), candidate(1, 0), candidate(2, 1)];
    assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 1);
    pool[1].health.healthy = false;
    assert_eq!(
        place_keyed(&engine, &pool, 7, 20).position,
        2,
        "the failover"
    );
    pool[1].health.healthy = true;
    assert_eq!(
        place_keyed(&engine, &pool, 7, 30).position,
        1,
        "the first owner leads under original recovery"
    );
    pool[1].session_in_flight = 2;
    assert_eq!(
        place_keyed(&engine, &pool, 7, 40).position,
        1,
        "the fewest requests are the owner's: the stages pick it again"
    );
    pool[1].in_flight = 5;
    assert_eq!(place_keyed(&engine, &pool, 7, 45).position, 2);
    pool[1].session_in_flight = 0;
    assert_eq!(
        place_keyed(&engine, &pool, 7, 50).position,
        2,
        "the respill moved the pin to the failover"
    );
}

/// At the cap the stages place the request over every candidate. A pick
/// on another worker at the cap takes the request and leaves the pin;
/// a pick on another worker under the cap takes the pin too.
#[test]
fn at_the_cap_a_request_goes_where_the_stages_say_and_the_pin_moves_only_under_the_cap() {
    let engine = session_engine(SESSIONS);
    let mut pool = vec![candidate(0, 0), candidate(1, 3), candidate(2, 3)];
    assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
    for c in &mut pool {
        c.session_in_flight = 2;
    }
    assert_eq!(
        place_keyed(&engine, &pool, 7, 20).position,
        0,
        "the stages pick the owner again"
    );
    // The fewest requests are now on a worker at the cap.
    pool[0].in_flight = 9;
    pool[1].in_flight = 2;
    let decision = decide_keyed(&engine, &pool, 7, 30);
    let pick = decision.pick().expect("the stages place");
    assert_eq!(pick.targets.first().map(|t| t.position), Some(1));
    assert_eq!(pick.session_verdict, SessionVerdict::None, "no pin moves");
    assert_eq!(place_keyed(&engine, &pool, 7, 40).position, 1);
    for c in &mut pool {
        c.session_in_flight = 0;
    }
    assert_eq!(
        place_keyed(&engine, &pool, 7, 50).position,
        0,
        "the pin stood on the owner"
    );
    // The owner at the cap again, the stages' pick under it.
    pool[0].session_in_flight = 2;
    assert_eq!(place_keyed(&engine, &pool, 7, 60).position, 1);
    pool[0].session_in_flight = 0;
    pool[0].in_flight = 0;
    assert_eq!(
        place_keyed(&engine, &pool, 7, 70).position,
        1,
        "the respill moved the pin"
    );
}

/// Who placed a keyed request's pick, at each place the whole decision
/// answers under a sessions rule. A key with no pin and a pin at its cap
/// are placed by the profile's stages under `delegate` and by the rule's
/// own stage under its own assignment. A pin followed is the rule's
/// under both.
#[test]
fn a_keyed_pick_says_who_placed_it() {
    use decision::PlacedBy;
    for (assignment, placed) in [
        (Assignment::Delegate, PlacedBy::Stages),
        (Assignment::MinLoad, PlacedBy::Rule),
    ] {
        let engine = session_engine(SessionRule {
            assignment,
            ..SESSIONS
        });
        let mut pool = vec![candidate(0, 0), candidate(1, 3), candidate(2, 3)];
        let by = |decision: &Decision| decision.pick().expect("a pick").placed_by();
        let new = decide_keyed(&engine, &pool, 7, 10);
        assert_eq!(by(&new), placed, "{assignment:?}: a key with no pin");
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
        let pinned = decide_keyed(&engine, &pool, 7, 20);
        assert_eq!(by(&pinned), PlacedBy::Rule, "{assignment:?}: a pin");
        // The pin at its cap and the busiest: the request is placed
        // again, on a worker under the cap.
        pool[0].session_in_flight = 2;
        pool[0].in_flight = 9;
        let again = decide_keyed(&engine, &pool, 7, 30);
        let pick = again.pick().expect("the respill places");
        assert_eq!(pick.session_verdict, SessionVerdict::Respill);
        assert_eq!(by(&again), placed, "{assignment:?}: a pin at its cap");
    }
}

/// A respill's commit fronts the worker it lands on, the first target or
/// an alternate behind it. So no worker at the cap stands among a
/// respill's targets: the pinned worker and a peer at the cap leave the
/// list, and an alternate's dispatch moves the pin under the cap. The
/// alternate then leads, and the worker the decision first took, which
/// it fronted where it was made, stands behind it.
#[test]
fn a_respill_names_no_alternate_at_the_cap() {
    let engine = session_engine(SESSIONS);
    let mut pool = vec![
        candidate(0, 0),
        candidate(1, 1),
        candidate(2, 2),
        candidate(3, 3),
    ];
    assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
    // The pin at the cap and the busiest; the stages' second choice at
    // the cap too.
    pool[0].session_in_flight = 2;
    pool[0].in_flight = 9;
    pool[2].session_in_flight = 2;
    let decision = decide_keyed(&engine, &pool, 7, 20);
    let pick = decision.pick().expect("the respill places");
    assert_eq!(pick.session_verdict, SessionVerdict::Respill);
    let placed: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
    assert_eq!(placed, [1, 3], "the workers under the cap alone");
    // The alternate's dispatch, under the decision's verdict.
    let alternate = pick.targets.as_slice()[1];
    let lease = Lease {
        session: session(7),
        session_verdict: pick.session_verdict,
        issued_micros: 20,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &alternate, Side::Single, 0);
    let first = pick.targets.as_slice()[0];
    assert_eq!(
        engine.sessions().pins(session(7)),
        Some([alternate.id, first.id]),
        "the alternate, then the worker first taken"
    );
    pool[0].session_in_flight = 0;
    assert_eq!(
        place_keyed(&engine, &pool, 7, 30).position,
        3,
        "the pin moved under the cap"
    );
}

/// A respill fronts the worker it takes where it is decided, ahead of
/// its dispatch: the key's next request follows the move under the cap,
/// as a gateway's sticky map moves a pin where it selects the worker.
/// The commit on that worker finds the pins in place.
#[test]
fn a_respill_fronts_its_worker_where_it_is_decided() {
    let engine = session_engine(SESSIONS);
    let mut pool = vec![candidate(0, 0), candidate(1, 1), candidate(2, 2)];
    let home = place_keyed(&engine, &pool, 7, 10);
    assert_eq!(home.position, 0);
    let pins = || engine.sessions().pins(session(7));
    // The pin at the cap and the busiest, and every other worker at the
    // cap too: the placement carries no verdict and moves no pin.
    pool[0].in_flight = 9;
    for row in &mut pool {
        row.session_in_flight = 2;
    }
    let over = decide_keyed(&engine, &pool, 7, 15);
    let pick = over.pick().expect("a placement at the cap");
    assert_eq!(pick.session_verdict, SessionVerdict::None);
    assert_eq!(pins(), Some([home.id, WorkerId(0)]), "no verdict");
    // The others under the cap: the respill takes the second row.
    (pool[1].session_in_flight, pool[2].session_in_flight) = (0, 0);
    let respill = decide_keyed(&engine, &pool, 7, 20);
    let pick = respill.pick().expect("the respill places");
    assert_eq!(pick.session_verdict, SessionVerdict::Respill);
    let taken = *pick.targets.first().expect("a target");
    assert_eq!(taken.position, 1);
    assert_eq!(pins(), Some([taken.id, home.id]), "ahead of its dispatch");
    // The key's next request, decided ahead of that dispatch.
    let next = decide_keyed(&engine, &pool, 7, 30);
    let pick = next.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
    assert_eq!(pick.targets.first().map(|target| target.id), Some(taken.id));
    let lease = Lease {
        session: session(7),
        session_verdict: SessionVerdict::Respill,
        issued_micros: 20,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &taken, Side::Single, 0);
    assert_eq!(pins(), Some([taken.id, home.id]), "behind its dispatch");
}

/// A lease's word takes back the claim its own decision made, and no later
/// decision's claim on the same worker. A key's claim idles out, and a
/// later decision claims the same worker. The older lease's refusal and
/// its dispatch elsewhere, both told late, leave that claim. The later
/// lease's own refusal takes it back. Each lease is dated by its
/// decision's instant, as a host dates it.
#[test]
fn a_leases_word_takes_back_the_claim_of_its_own_decision_alone() {
    let engine = session_engine(SessionRule {
        ttl_micros: Some(1_000),
        sweep_micros: 100,
        ..SESSIONS
    });
    // The fewest requests place the key on the first worker each time.
    let pool = [candidate(0, 0), candidate(1, 5)];
    let (claimed, other) = (pool[0].id, pool[1].id);
    let pins = || engine.sessions().pins(session(7));
    let leased = |now: u64| {
        let decision = decide_keyed(&engine, &pool, 7, now);
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Claimed(claimed));
        Lease {
            session: session(7),
            session_verdict: pick.session_verdict,
            issued_micros: now,
            ..Lease::default()
        }
    };
    // The key idles out between the two decisions.
    let (older, later) = (leased(10), leased(1_200));
    assert_eq!(pins(), Some([claimed, WorkerId(0)]));
    engine.on_released(&older, ReleaseReason::CommitFailed);
    assert_eq!(pins(), Some([claimed, WorkerId(0)]), "the later claim");
    let elsewhere = Target {
        position: 1,
        id: other,
        ..Target::default()
    };
    engine.on_dispatched(&older, &elsewhere, Side::Single, 0);
    assert_eq!(pins(), Some([other, claimed]), "the later claim stands");
    engine.on_released(&later, ReleaseReason::CommitFailed);
    assert_eq!(pins(), Some([other, WorkerId(0)]), "its own refusal");
    // A lease that states no instant reaches a claim by its worker.
    let undated = Lease {
        issued_micros: 0,
        ..leased(2_400)
    };
    assert_eq!(pins(), Some([claimed, WorkerId(0)]), "the key started over");
    engine.on_released(&undated, ReleaseReason::CommitFailed);
    assert_eq!(pins(), Some([WorkerId(0); 2]), "by its worker");
}

/// The pin taken again at the cap is the decision's one target. A
/// dispatch on an alternate would move the pin by the key's recovery,
/// onto a worker at the cap as onto any other.
#[test]
fn the_pin_taken_again_at_the_cap_names_no_alternate() {
    let engine = session_engine(SESSIONS);
    let mut pool = vec![
        candidate(0, 0),
        candidate(1, 1),
        candidate(2, 2),
        candidate(3, 3),
    ];
    assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
    // Every worker at the cap, the pin with the fewest requests: the
    // stages pick it again.
    for c in &mut pool {
        c.session_in_flight = 2;
    }
    let decision = decide_keyed(&engine, &pool, 7, 20);
    let pick = decision.pick().expect("the stages place");
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
    // The dispatch of the decision's last target, under its verdict.
    let last = *pick.targets.as_slice().last().expect("a target");
    let lease = Lease {
        session: session(7),
        session_verdict: pick.session_verdict,
        issued_micros: 20,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &last, Side::Single, 0);
    for c in &mut pool {
        c.session_in_flight = 0;
    }
    assert_eq!(
        place_keyed(&engine, &pool, 7, 30).position,
        0,
        "the pin stayed"
    );
    let placed: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
    assert_eq!(placed, [0], "the pin alone");
}

/// A key with no routable pin is placed by the rule's assignment: the
/// profile's own stages under `delegate`, the fewest requests under
/// `min_load`, the fewest sessions under `min_group`, any routable worker
/// under `random`.
#[test]
fn a_key_without_a_routable_pin_is_placed_by_the_rules_assignment() {
    let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
    pool[0].active_sessions = 7;
    pool[1].active_sessions = 1;
    pool[2].active_sessions = 3;
    let by = |assignment| {
        let engine = session_engine(SessionRule {
            assignment,
            ..SESSIONS
        });
        place_keyed(&engine, &pool, 7, 10).position
    };
    assert_eq!(by(Assignment::Delegate), 0, "the profile's fewest requests");
    assert_eq!(by(Assignment::MinLoad), 0);
    assert_eq!(by(Assignment::MinGroup), 1, "the fewest sessions");
    assert!(by(Assignment::Random) < 3);
}

/// A join under an id that left ends its absence, whoever tells it:
/// the join's own word, a replacement's new id, a snapshot that lists
/// the worker.
#[test]
fn a_join_under_an_id_that_left_ends_its_absence() {
    let engine = session_engine(SESSIONS);
    let pool = vec![candidate(0, 1), candidate(1, 4)];
    // A key in the table: a table that holds none files no absence.
    place_keyed(&engine, &pool, 7, 10);
    let (a, b) = (WorkerId(100), WorkerId(101));
    engine.on_membership(&Membership::Removed(a));
    assert_eq!(engine.sessions.absent(), 1);
    engine.on_membership(&Membership::Added(a));
    assert_eq!(engine.sessions.absent(), 0, "a join's own word");
    engine.on_membership(&Membership::Removed(a));
    engine.on_membership(&Membership::Removed(b));
    engine.on_membership(&Membership::Snapshot(&[a]));
    assert_eq!(engine.sessions.absent(), 1, "the snapshot lists one");
    let replaced = Membership::Replaced {
        old: WorkerId(300),
        new: b,
    };
    engine.on_membership(&replaced);
    assert_eq!(engine.sessions.absent(), 1, "the old life alone");
}

#[test]
fn a_removed_workers_keys_fall_to_their_failover_and_go_back_with_the_worker() {
    let engine = session_engine(SESSIONS);
    let mut pool = vec![
        candidate(0, 1),
        candidate(1, 4),
        candidate(2, 2),
        candidate(3, 3),
    ];
    // Key 7 lives on worker 100 alone; key 8 moves to 102 while 100 is
    // down and keeps 100 as its failover.
    assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
    assert_eq!(place_keyed(&engine, &pool, 8, 10).position, 0);
    pool[0].health.healthy = false;
    assert_eq!(place_keyed(&engine, &pool, 8, 20).position, 2);
    pool[0].health.healthy = true;
    pool[0].in_flight = 9;
    assert_eq!(
        place_keyed(&engine, &pool, 7, 30).position,
        0,
        "a pin holds under load"
    );
    assert_eq!(place_keyed(&engine, &pool, 8, 30).position, 2);
    // A host that drops a row numbers the rest again: a row's position
    // is its index in the pool a decision is given.
    let renumber = |pool: &mut [Candidate]| {
        for (row, index) in pool.iter_mut().zip(0u32..) {
            row.position = index;
        }
    };
    // The host removes 102: key 8's pins stand through the leave, its
    // owner reads as not routable and its failover serves, loaded or
    // not.
    engine.on_membership(&Membership::Removed(WorkerId(102)));
    pool.remove(2);
    renumber(&mut pool);
    assert_eq!(
        engine.sessions.pins(session(8)),
        Some([WorkerId(102), WorkerId(100)]),
        "the pins stand through the leave"
    );
    assert_eq!(place_keyed(&engine, &pool, 8, 40).id, WorkerId(100));
    // The host registers the key of 102 again, as the worker it was: a
    // new id, and its pins with it.
    engine.on_membership(&Membership::Added(WorkerId(202)));
    engine.pins_follow(WorkerId(102), WorkerId(202));
    pool.push(Candidate {
        id: WorkerId(202),
        ..candidate(3, 5)
    });
    assert_eq!(
        engine.sessions.pins(session(8)),
        Some([WorkerId(202), WorkerId(100)]),
        "the pins follow the key"
    );
    assert_eq!(place_keyed(&engine, &pool, 8, 50).id, WorkerId(202));
    // The host replaces 100 and says the new life is another worker:
    // key 7 has no pin left and is placed afresh by the rule's
    // assignment (the fewest requests), and key 8 keeps its owner.
    engine.on_membership(&Membership::Replaced {
        old: WorkerId(100),
        new: WorkerId(200),
    });
    assert_eq!(
        engine.sessions.pins(session(7)),
        Some([WorkerId(100), WorkerId(0)]),
        "a replacement's pins stand until the host says which way"
    );
    engine.pins_forget(&[WorkerId(100)]);
    pool.remove(0);
    renumber(&mut pool);
    assert_eq!(engine.sessions.pins(session(7)), None);
    assert_eq!(
        engine.sessions.pins(session(8)),
        Some([WorkerId(202), WorkerId(0)])
    );
    assert_eq!(place_keyed(&engine, &pool, 7, 60).id, WorkerId(103));
    assert_eq!(place_keyed(&engine, &pool, 8, 60).id, WorkerId(202));
}

/// A decision reads its rows ahead of its claim. Where its worker is
/// registered again between the two, the key is claimed under the id
/// that stands: the key's next request, decided on the rows that stand
/// ahead of the first one's dispatch, follows the claim. The first
/// request's verdict and its dispatch name the id its decision read,
/// and the dispatch confirms the claim.
#[test]
fn a_claim_behind_its_workers_rename_leads_under_the_id_that_stands() {
    let engine = session_engine(SESSIONS);
    // The rows the first decision read, and the rows that stand: the
    // host replaced the first worker's row and says it is the worker
    // it was.
    let read = [candidate(0, 0), candidate(1, 1)];
    let mut rows = read;
    rows[0].id = WorkerId(900);
    engine.on_membership(&Membership::Replaced {
        old: read[0].id,
        new: rows[0].id,
    });
    engine.pins_follow(read[0].id, rows[0].id);
    let decision = decide_keyed(&engine, &read, 7, 10);
    let pick = decision.pick().expect("a placement");
    let first = *pick.targets.first().expect("a target");
    assert_eq!(first.id, read[0].id);
    assert_eq!(pick.session_verdict, SessionVerdict::Claimed(read[0].id));
    let pins = || engine.sessions.pins(session(7));
    assert_eq!(
        pins(),
        Some([rows[0].id, WorkerId(0)]),
        "the id that stands"
    );
    // The worker is the busiest now: the key's pin alone sends there.
    rows[0].in_flight = 9;
    let next = decide_keyed(&engine, &rows, 7, 20);
    let pick = next.pick().expect("a pick");
    assert_eq!(pick.targets.first().map(|t| t.id), Some(rows[0].id));
    let lease = Lease {
        session: session(7),
        session_verdict: SessionVerdict::Claimed(read[0].id),
        issued_micros: 10,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &first, Side::Single, 0);
    assert_eq!(pins(), Some([rows[0].id, WorkerId(0)]), "confirmed");
}

/// A key pinned on a worker and decided on rows read before the worker
/// was registered again: the pin names the id that stands, no row
/// read holds it, and the rotation's turn lands on the worker under
/// the id it had. The table answers that the worker is a pin of the
/// key already, and the decision takes its pick as the pin it is:
/// nothing is claimed, and the rule's cap holds for it. At the cap the
/// key is handed on to the rotation's next turn, as from a pin read
/// first, and that worker leads the pins ahead of the dispatch.
#[test]
fn a_pick_answered_as_a_pin_already_is_put_to_the_rules_cap() {
    let engine = session_engine_of(round_robin_profile(), SESSIONS);
    let read = [
        candidate(0, 0),
        candidate(1, 0),
        candidate(2, 0),
        candidate(3, 0),
    ];
    let home = place_keyed(&engine, &read, 7, 10);
    let stands = WorkerId(900);
    engine.on_membership(&Membership::Replaced {
        old: home.id,
        new: stands,
    });
    engine.pins_follow(home.id, stands);
    let pins = || engine.sessions.pins(session(7));
    assert_eq!(pins(), Some([stands, WorkerId(0)]));
    // A request with no key takes one turn of the rotation.
    let turn = |now: u64| {
        let facts = RequestFacts {
            model: "m",
            now_micros: now,
            ..RequestFacts::default()
        };
        let (shared, cx) = (PoolFacts::default(), Constraints::default());
        let decision = engine.decide(&facts, &shared, &read, &cx, &mut Scratch::new());
        let pick = decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    // Three turns bring the rotation back to the key's worker. Under
    // the cap its turn is the pin: no claim, and nothing is written.
    let behind: Vec<u32> = (0..3).map(|step| turn(20 + step)).collect();
    assert!(!behind.contains(&home.position), "{behind:?}");
    let under = decide_keyed(&engine, &read, 7, 30);
    let pick = under.pick().expect("a pick");
    assert_eq!(pick.targets.first().map(|t| t.id), Some(home.id));
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned, "no claim");
    assert_eq!(pins(), Some([stands, WorkerId(0)]), "nothing is written");
    // At the cap, a rotation later: the key is handed on.
    let behind: Vec<u32> = (0..3).map(|step| turn(40 + step)).collect();
    assert!(!behind.contains(&home.position), "{behind:?}");
    let mut capped = read;
    capped[home.position as usize].session_in_flight = SESSIONS.inflight_cap;
    let handed = decide_keyed(&engine, &capped, 7, 50);
    let pick = handed.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::Respill);
    let taken = *pick.targets.first().expect("a target");
    assert_ne!(taken.id, home.id);
    assert_eq!(pins(), Some([taken.id, stands]), "ahead of its dispatch");
}

#[test]
fn two_first_decisions_of_a_fresh_key_agree_on_the_first_claim() {
    let engine = session_engine(SESSIONS);
    let key = session(7);
    let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
    let first = decide_keyed(&engine, &pool, 7, 10);
    let first_pick = first.pick().expect("a pick");
    assert_eq!(first_pick.targets.first().map(|t| t.position), Some(0));
    let claimed = first_pick.targets.first().expect("a target").id;
    assert_eq!(first_pick.session_verdict, SessionVerdict::Claimed(claimed));
    // Before the first commits, a second decision under a changed load
    // would place the key elsewhere; it reads the claim as the pin and
    // takes it as a hit. (Between a decision's own reading and its claim
    // another claim is followed the same way, when among its survivors.)
    pool[0].in_flight = 9;
    let second = decide_keyed(&engine, &pool, 7, 11);
    let targets = second.pick().expect("a pick").targets;
    assert_eq!(
        targets.first().map(|t| t.position),
        Some(0),
        "the claim leads"
    );
    assert_eq!(targets.len(), 1, "a hit on the claim");
    assert_eq!(
        engine.sessions.pins(key),
        Some([WorkerId(100), WorkerId(0)])
    );
    // The commits, in either order, keep the one pin.
    let lease = |decision: &Decision, now| Lease {
        session: key,
        session_verdict: decision.pick().expect("a pick").session_verdict,
        issued_micros: now,
        ..Lease::default()
    };
    let target = |decision: &Decision| {
        *decision
            .pick()
            .expect("a pick")
            .targets
            .first()
            .expect("a target")
    };
    engine.on_dispatched(&lease(&second, 11), &target(&second), Side::Single, 0);
    engine.on_dispatched(&lease(&first, 10), &target(&first), Side::Single, 0);
    assert_eq!(
        engine.sessions.pins(key),
        Some([WorkerId(100), WorkerId(0)])
    );
}

/// The owner is down; the decision claims the best survivor, the commit
/// refuses it and dispatches the alternate: the owner keeps its pin, the
/// alternate is the failover, and the owner leads again once it is back.
#[test]
fn an_alternate_commit_withdraws_the_claim_and_keeps_the_owner() {
    let engine = session_engine(SessionRule {
        recovery: Recovery::Original,
        ..SESSIONS
    });
    let key = session(7);
    let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
    let owner = place_keyed(&engine, &pool, 7, 10);
    assert_eq!(owner.position, 0);
    pool[0].health.healthy = false;
    let decision = decide_keyed(&engine, &pool, 7, 20);
    let pick = decision.pick().expect("a pick");
    let claimed = pick.targets.first().expect("a target").id;
    let alternate = pick.targets.as_slice()[1];
    assert_ne!(alternate.id, claimed);
    assert_eq!(engine.sessions.pins(key), Some([owner.id, claimed]));
    let lease = Lease {
        session: key,
        session_verdict: pick.session_verdict,
        issued_micros: 20,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &alternate, Side::Single, 0);
    assert_eq!(
        engine.sessions.pins(key),
        Some([owner.id, alternate.id]),
        "the claim is withdrawn, the owner keeps the lead"
    );
    pool[0].health.healthy = true;
    assert_eq!(place_keyed(&engine, &pool, 7, 30).position, 0);
}

/// Two bindings' rules share the engine's table, which sweeps at the
/// shorter of their intervals: once the short rule's TTL and interval
/// passed, its idle key is gone from the table while the hourly key
/// stays, whichever binding's decision ran the sweep.
#[test]
fn the_table_sweeps_at_the_shortest_interval_of_the_bindings_rules() {
    let hourly = SessionRule {
        ttl_micros: Some(3_600_000_000),
        sweep_micros: 3_600_000_000,
        ..SESSIONS
    };
    let short = SessionRule {
        ttl_micros: Some(1_000),
        sweep_micros: 1_000,
        ..SESSIONS
    };
    let binding = |model: &str, rule: SessionRule| Binding {
        model: model.into(),
        phases: vec![],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow: None,
        decode_shadow: None,
        sessions: Some(rule),
    };
    let engine = Engine::new(
        vec![least_requests_profile()],
        vec![binding("a", hourly), binding("b", short)],
        1,
        Needs::default(),
    );
    let pool = vec![candidate(0, 1), candidate(1, 4)];
    let place = |model: &'static str, key: u64, now: u64| {
        let facts = RequestFacts {
            model,
            session_key: Identity::hashed(key),
            now_micros: now,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a pick");
        let target = *pick.targets.first().expect("a target");
        let lease = Lease {
            session: facts.session(),
            session_verdict: pick.session_verdict,
            issued_micros: now,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
    };
    place("a", 1, 0);
    place("b", 2, 0);
    place("b", 3, 2_000);
    assert_eq!(
        engine.sessions().len(),
        2,
        "the hourly key and the fresh one stay, the short key was swept"
    );
    let swept = RequestFacts {
        model: "b",
        session_key: Identity::hashed(2),
        ..RequestFacts::default()
    };
    assert!(engine.sessions.pins(swept.session()).is_none());
}

#[test]
fn a_key_whose_pins_are_all_down_is_claimed_at_its_decision_too() {
    let engine = session_engine(SESSIONS);
    let key = session(7);
    let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
    assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
    pool[0].health.healthy = false;
    // The owner is down: the first decision claims the fewest-requests
    // worker before any commit; a second decision under a changed load
    // reads that claim as the pin and takes it.
    let first = decide_keyed(&engine, &pool, 7, 20);
    let lead = |d: &Decision| {
        d.pick()
            .expect("a pick")
            .targets
            .first()
            .map(|t| t.position)
    };
    assert_eq!(lead(&first), Some(2));
    assert_eq!(
        engine.sessions.pins(key),
        Some([WorkerId(102), WorkerId(100)])
    );
    pool[2].in_flight = 9;
    let second = decide_keyed(&engine, &pool, 7, 21);
    assert_eq!(lead(&second), Some(2), "the claim is the pin");
}

#[test]
fn a_keyed_request_under_a_binding_without_a_sessions_rule_pins_nothing() {
    let engine = Engine::new(
        vec![least_requests_profile()],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let pool = vec![candidate(0, 1), candidate(1, 4)];
    let decision = decide_keyed(&engine, &pool, 7, 10);
    let pick = decision.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::None);
    let lease = Lease {
        session: session(7),
        session_verdict: pick.session_verdict,
        issued_micros: 10,
        ..Lease::default()
    };
    engine.on_dispatched(
        &lease,
        pick.targets.first().expect("a target"),
        Side::Single,
        0,
    );
    assert!(engine.sessions.is_empty(), "no rule, no row");
}

/// One engine, two hosts' requests for one key: the first holds the
/// plain key, the second its fingerprint alone. Built apart, they name
/// one session, so the second meets the first's pin past the load; the
/// key under another model, or in another phase, is another session and
/// is placed afresh.
#[test]
fn two_hosts_requests_for_one_key_model_and_phase_meet_on_one_pin() {
    let engine = session_engine(SESSIONS);
    let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
    let commit = |facts: &RequestFacts<'_>, pool: &[Candidate]| {
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a pick");
        let target = *pick.targets.first().expect("a target");
        let lease = Lease {
            session: facts.session(),
            session_verdict: pick.session_verdict,
            issued_micros: facts.now_micros,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
        (target.position, pick.session_verdict)
    };
    let plain = RequestFacts {
        model: "m",
        session_key: Identity::plain("conversation-7"),
        now_micros: 10,
        ..RequestFacts::default()
    };
    let (first, verdict) = commit(&plain, &pool);
    assert_eq!(first, 0, "the fewest requests");
    assert_eq!(verdict, SessionVerdict::Claimed(WorkerId(100)));
    // The other host holds the key's fingerprint alone, and the pinned
    // worker is the busiest by now.
    pool[0].in_flight = 9;
    let hashed = RequestFacts {
        session_key: Identity::hashed(view::fingerprint(b"conversation-7")),
        now_micros: 20,
        ..plain
    };
    assert_eq!(
        commit(&hashed, &pool),
        (0, SessionVerdict::Pinned),
        "one pin for both forms of the key"
    );
    assert_eq!(engine.sessions().len(), 1);
    let other_model = RequestFacts {
        model: "n",
        now_micros: 30,
        ..plain
    };
    assert_eq!(
        commit(&other_model, &pool).0,
        2,
        "placed afresh, by the load"
    );
    let other_phase = RequestFacts {
        phase: Phase::Decode,
        now_micros: 40,
        ..plain
    };
    assert_eq!(
        commit(&other_phase, &pool).0,
        2,
        "placed afresh, by the load"
    );
    assert_eq!(engine.sessions().len(), 3, "three sessions of one key");
}

fn manual_engine(sessions: Option<SessionRule>) -> Engine {
    Engine::new(
        vec![manual_profile()],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions,
        }],
        1,
        Needs::default(),
    )
}

/// One decision for `key` (`0`: no key) drawn by `seed`: the worker and
/// what the decision did with the key.
fn decide_seeded(
    engine: &Engine,
    pool: &[Candidate],
    key: u64,
    seed: u64,
) -> (u32, SessionVerdict) {
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::hashed(key),
        seed,
        now_micros: 10,
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &facts,
        &PoolFacts::default(),
        pool,
        &Constraints::default(),
        &mut scratch,
    );
    let pick = decision.pick().expect("a pick");
    (
        pick.targets.first().expect("a target").position,
        pick.session_verdict,
    )
}

/// Under the `manual` picker the rule's assignment places a keyless
/// request and a key with no pin alike, through one stage: the two
/// agree pick for pick on every seed. Without a key nothing is pinned.
#[test]
fn the_manual_picker_places_a_keyed_miss_and_a_keyless_request_by_one_assignment() {
    let mut pool = vec![
        candidate(0, 1),
        candidate(1, 4),
        candidate(2, 1),
        candidate(3, 2),
    ];
    pool[0].active_sessions = 7;
    pool[1].active_sessions = 1;
    pool[2].active_sessions = 3;
    pool[3].active_sessions = 1;
    // The workers an assignment may take: any, the fewest requests in
    // flight, the fewest session keys.
    for (assignment, may) in [
        (Assignment::Random, &[0u32, 1, 2, 3][..]),
        (Assignment::MinLoad, &[0, 2][..]),
        (Assignment::MinGroup, &[1, 3][..]),
        // No stages of its own to delegate to: the fewest in flight.
        (Assignment::Delegate, &[0, 2][..]),
    ] {
        let rule = SessionRule {
            assignment,
            ..SESSIONS
        };
        let mut taken = [false; 4];
        for seed in 1..=64u64 {
            let (keyless, verdict) = decide_seeded(&manual_engine(Some(rule)), &pool, 0, seed);
            assert_eq!(verdict, SessionVerdict::None, "{assignment:?}");
            let (keyed, verdict) = decide_seeded(&manual_engine(Some(rule)), &pool, 7, seed);
            assert!(
                matches!(verdict, SessionVerdict::Claimed(_)),
                "{assignment:?}: {verdict:?}"
            );
            assert_eq!(keyless, keyed, "{assignment:?}, seed {seed}");
            assert!(may.contains(&keyless), "{assignment:?} took {keyless}");
            taken[keyless as usize] = true;
        }
        for &position in may {
            assert!(
                taken[position as usize],
                "{assignment:?} never took {position}"
            );
        }
    }
}

#[test]
fn a_keyless_request_under_the_manual_picker_pins_nothing() {
    let engine = manual_engine(Some(SESSIONS));
    let pool = vec![candidate(0, 3), candidate(1, 1)];
    let target = place_keyed(&engine, &pool, 0, 10);
    assert_eq!(target.position, 1, "the fewest in flight under delegate");
    assert!(engine.sessions.is_empty(), "no key, no row");
}

/// A binding with no sessions rule names no assignment: the picker
/// takes the fewest in flight, with a key or without, and pins nothing.
#[test]
fn without_a_rule_the_manual_picker_takes_the_fewest_in_flight() {
    let engine = manual_engine(None);
    let pool = vec![candidate(0, 3), candidate(1, 1), candidate(2, 2)];
    for key in [0, 7] {
        for seed in 1..=8 {
            assert_eq!(
                decide_seeded(&engine, &pool, key, seed),
                (1, SessionVerdict::None)
            );
        }
    }
    assert!(engine.sessions.is_empty());
}

/// A reader reads back the document: the binding the engine answers
/// holds its rule as it was handed in, under every assignment and under
/// none. Under `delegate` the `manual` picker's requests still go to
/// the fewest in flight: the assignment that places is the engine's own
/// and no reader of the binding sees it.
#[test]
fn the_engine_answers_a_binding_as_it_was_declared() {
    for assignment in [
        Assignment::Random,
        Assignment::MinLoad,
        Assignment::MinGroup,
        Assignment::Delegate,
    ] {
        let rule = SessionRule {
            assignment,
            ..SESSIONS
        };
        let engine = manual_engine(Some(rule));
        let read_back = engine
            .binding("m", Phase::Single)
            .and_then(|binding| binding.sessions);
        assert_eq!(read_back, Some(rule), "{assignment:?}");
    }
    let unruled = manual_engine(None);
    let read_back = unruled
        .binding("m", Phase::Single)
        .map(|binding| binding.sessions);
    assert_eq!(read_back, Some(None), "no rule declared, none read back");
    // One engine, both halves: the rule reads back `delegate` and the
    // request is placed by the fewest in flight.
    let engine = manual_engine(Some(SESSIONS));
    let declared = engine
        .binding("m", Phase::Single)
        .and_then(|binding| binding.sessions)
        .map(|rule| rule.assignment);
    assert_eq!(declared, Some(Assignment::Delegate));
    let pool = vec![candidate(0, 3), candidate(1, 1)];
    assert_eq!(place_keyed(&engine, &pool, 0, 10).position, 1);
}

/// The keyed path under the `manual` picker is the engine's own: the
/// key stays on its worker past the load, moves when the worker is
/// unroutable, and the first owner leads again once it is back.
#[test]
fn a_key_under_the_manual_picker_stays_on_its_worker_and_the_first_owner_leads_again() {
    let engine = manual_engine(Some(SessionRule {
        assignment: Assignment::MinLoad,
        recovery: Recovery::Original,
        ..SESSIONS
    }));
    let mut pool = vec![candidate(0, 5), candidate(1, 0), candidate(2, 3)];
    assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 1);
    pool[1].in_flight = 9;
    assert_eq!(place_keyed(&engine, &pool, 7, 20).position, 1);
    pool[1].health.healthy = false;
    assert_eq!(
        place_keyed(&engine, &pool, 7, 30).position,
        2,
        "the fewest in flight of the routable workers"
    );
    pool[1].health.healthy = true;
    assert_eq!(place_keyed(&engine, &pool, 7, 40).position, 1);
    assert_eq!(
        engine.sessions.pins(session(7)),
        Some([WorkerId(101), WorkerId(102)])
    );
}

/// A pin over the rule's in-flight cap respills by the binding's placing
/// assignment too: under the `manual` picker a `delegate` rule hands the
/// key to the fewest in flight of the candidates, and the decision says
/// a respill where that is another worker under the cap.
#[test]
fn a_respill_under_the_manual_picker_takes_the_fewest_in_flight() {
    let engine = manual_engine(Some(SESSIONS));
    let mut pool = vec![candidate(0, 5), candidate(1, 0), candidate(2, 3)];
    assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 1);
    pool[1].session_in_flight = SESSIONS.inflight_cap;
    pool[1].in_flight = 9;
    let decision = decide_keyed(&engine, &pool, 7, 20);
    let pick = decision.pick().expect("the respill places");
    assert_eq!(pick.session_verdict, SessionVerdict::Respill);
    assert_eq!(pick.targets.first().map(|t| t.position), Some(2));
}

/// The profile's filters narrow the pool before the assignment places a
/// request with no key: a worker that is down takes none under any
/// assignment, or with no rule, though it holds the fewest requests and
/// the fewest keys.
#[test]
fn a_keyless_request_under_the_manual_picker_passes_the_profiles_filters() {
    let mut pool = vec![candidate(0, 0), candidate(1, 3), candidate(2, 2)];
    pool[0].health.healthy = false;
    pool[1].active_sessions = 5;
    pool[2].active_sessions = 4;
    let rules = [
        Assignment::Random,
        Assignment::MinLoad,
        Assignment::MinGroup,
        Assignment::Delegate,
    ]
    .map(|assignment| {
        Some(SessionRule {
            assignment,
            ..SESSIONS
        })
    });
    for rule in rules.into_iter().chain([None]) {
        let engine = manual_engine(rule);
        for seed in 1..=64u64 {
            let (taken, _) = decide_seeded(&engine, &pool, 0, seed);
            assert_ne!(taken, 0, "{rule:?}, seed {seed}: the worker that is down");
        }
    }
}

/// A lease takes its session from the request and its verdict from the
/// pick, for every keyed request: under a sessions rule the dispatch
/// pins by the two, and with no rule the same lease moves no pin, the
/// verdict alone deciding it.
#[test]
fn a_lease_takes_the_requests_session_and_the_picks_verdict() {
    let pool = vec![candidate(0, 3), candidate(1, 1)];
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::hashed(7),
        now_micros: 10,
        ..RequestFacts::default()
    };
    let pick_of = |engine: &Engine| {
        let mut scratch = Scratch::new();
        *engine
            .decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut scratch,
            )
            .pick()
            .expect("a pick")
    };
    let lease_of = |pick: &Pick| Lease {
        session: facts.session(),
        session_verdict: pick.session_verdict,
        issued_micros: 10,
        ..Lease::default()
    };
    let ruled = manual_engine(Some(SESSIONS));
    let miss = pick_of(&ruled);
    let target = *miss.targets.first().expect("a target");
    ruled.on_dispatched(&lease_of(&miss), &target, Side::Single, 0);
    let hit = pick_of(&ruled);
    assert_eq!(hit.session_verdict, SessionVerdict::Pinned);
    assert_eq!(hit.targets.first().map(|t| t.id), Some(target.id));
    let bare = manual_engine(None);
    let unruled = pick_of(&bare);
    assert_eq!(unruled.session_verdict, SessionVerdict::None);
    let target = *unruled.targets.first().expect("a target");
    bare.on_dispatched(&lease_of(&unruled), &target, Side::Single, 0);
    assert!(bare.sessions.is_empty(), "no rule, no row");
}
