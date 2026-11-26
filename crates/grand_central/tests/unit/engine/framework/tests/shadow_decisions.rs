use super::*;

fn with_shadow(shadow: Profile, sessions: Option<SessionRule>) -> Engine {
    Engine::new(
        vec![least_requests_profile(), shadow],
        vec![Binding {
            shadow: Some(1),
            sessions,
            ..single_binding()
        }],
        23,
        Needs::default(),
    )
}

fn by_sessions() -> Profile {
    Profile {
        name: "by-sessions".into(),
        scorers: vec![WeightedScorer {
            scorer: Box::new(FewestSessions),
            weight: 1.0,
        }],
        ..least_requests_profile()
    }
}

/// A keyed request of model `m` at `now`, its key's counts held.
fn keyed_at(key: u64, now: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        session_key: Identity::hashed(key),
        session_counted: true,
        now_micros: now,
        ..RequestFacts::default()
    }
}

/// The worker of a decision's first target.
fn first_of(decision: &Decision) -> Option<WorkerId> {
    decision
        .pick()
        .and_then(|pick| pick.targets.first())
        .map(|target| target.id)
}

/// Dispatch the target at `at` of a live decision, as a host commits it.
fn commit(engine: &Engine, facts: &RequestFacts<'_>, live: &Decision, at: usize) -> WorkerId {
    let pick = live.pick().expect("a pick");
    let target = pick.targets.as_slice()[at];
    let lease = Lease {
        session: facts.session(),
        session_verdict: pick.session_verdict,
        issued_micros: facts.now_micros,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &target, Side::Single, 0);
    target.id
}

/// A live profile that reads no ring states no place on one, and its
/// shadow is apart from none: a ring the request's caller lends is the
/// shadow's, under the plain ring and under the prefix form alike. The
/// shadow's first target is the worker that ring reaches first,
/// whichever it is.
#[test]
fn a_shadow_beside_a_live_profile_that_reads_no_ring_keeps_the_lent_ring() {
    use crate::policies::{
        cache_aware::Gates,
        consistent_hashing::{tests::InOrder, ConsistentHash},
    };

    let mut rows = [candidate(0, 3), candidate(1, 5), candidate(2, 9)];
    for (at, row) in rows.iter_mut().enumerate() {
        row.id = WorkerId::from_address(&format!("w{at}"));
    }
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    for prefixed in [false, true] {
        let ring = ConsistentHash::new(None);
        let ring = if prefixed {
            ring.keyed_by_prefix(256, Gates::default(), &[])
        } else {
            ring
        };
        let engine = with_shadow(
            Profile {
                name: "ring".into(),
                scorers: vec![WeightedScorer {
                    scorer: Box::new(ring),
                    weight: 1.0,
                }],
                ..least_requests_profile()
            },
            None,
        );
        for first in [2usize, 0] {
            let order = [first, 1, 2 - first].map(|at| format!("w{at}"));
            let lent = InOrder(order.to_vec());
            let facts = RequestFacts {
                model: "m",
                session_key: Identity::hashed(9),
                ring: Some(&lent),
                ring_positions: &[0],
                ..RequestFacts::default()
            };
            let shadow = engine
                .shadow(&facts, &pool, &rows, &cx, &mut scratch)
                .expect("a shadow decision");
            assert_eq!(
                first_of(&shadow),
                Some(rows[first].id),
                "prefixed: {prefixed}"
            );
        }
    }
}

#[test]
fn shadow_picks_with_its_own_profile_over_the_same_rows() {
    let engine = with_shadow(by_sessions(), None);
    let mut busy = candidate(1, 3);
    busy.active_sessions = 17;
    let rows = [candidate(0, 13), busy];
    let (facts, pool, cx) = (
        RequestFacts::default(),
        PoolFacts::default(),
        Constraints::default(),
    );
    let mut scratch = Scratch::new();
    let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    let first = |decision: &Decision| {
        decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .map(|t| t.id)
    };
    assert_eq!(first(&live), Some(rows[1].id));
    assert_eq!(first(&shadow), Some(rows[0].id));
    assert_eq!(shadow.trace().profile, 1);
    assert_eq!(
        decision::ShadowKind::of(&live, &shadow),
        decision::ShadowKind::OtherTarget
    );

    let malformed = RequestFacts {
        affinity: &[AffinityFacts::default()],
        ..facts
    };
    assert_eq!(
        engine.shadow(&malformed, &pool, &rows, &cx, &mut scratch),
        Some(engine.decide(&malformed, &pool, &rows, &cx, &mut scratch))
    );
}

/// A shadow is lent no word of the pool. Where the live profile and its
/// shadow both rotate, the live decisions take the list's rows one after
/// the other, and the word counts them alone, whatever the shadows
/// between them take.
#[test]
fn a_shadow_turns_no_rotation_of_the_live_list() {
    use std::sync::atomic::Ordering;

    let engine = Engine::new(
        vec![round_robin_profile(), round_robin_profile()],
        vec![Binding {
            shadow: Some(1),
            ..single_binding()
        }],
        23,
        Needs::default(),
    );
    let rows = pool_of(DRAW_FROM as u32, |_| false);
    let word = AtomicU64::new(3);
    let pool = PoolFacts {
        word: Some(&word),
        ..PoolFacts::default()
    };
    let (facts, cx) = (request(0), Constraints::default());
    let mut scratch = Scratch::new();
    let mut lives = Vec::new();
    for _ in 0..6 {
        let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
        lives.push(positions(&live)[0]);
        let shadow = engine.shadow(&facts, &pool, &rows, &cx, &mut scratch);
        assert!(shadow.expect("a shadow decision").pick().is_some());
    }
    assert_eq!(lives, [3, 4, 5, 6, 7, 8]);
    assert_eq!(word.load(Ordering::Relaxed), 9);
}

#[test]
fn shadow_uses_the_bindings_all_vetoed_rule() {
    let profile = Profile {
        filters: vec![
            Box::new(Healthy),
            Box::new(NotOverloaded),
            Box::new(NotTried),
        ],
        ..least_requests_profile()
    };
    let mut engine = with_shadow(profile, None);
    let mut rows = [candidate(0, 29), candidate(1, 7)];
    for row in &mut rows {
        row.health.overloaded = true;
    }
    let (facts, pool, cx) = (
        RequestFacts::default(),
        PoolFacts::default(),
        Constraints::default(),
    );
    let mut scratch = Scratch::new();
    let steered = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("the binding shadows");
    assert!(matches!(steered, Decision::Pick(pick) if pick.trace.lifted == LiftedVeto::Overload));
    engine.shed_when_all_vetoed(0);
    let shed = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("the binding shadows");
    assert!(matches!(shed, Decision::Shed(shed) if shed.reason == ShedReason::Saturated));
}

#[test]
fn shadow_places_by_its_pickers_assignment() {
    let engine = with_shadow(
        Profile {
            scorers: vec![],
            picker: Box::new(Manual),
            ..least_requests_profile()
        },
        Some(SESSIONS),
    );
    let rows = [candidate(0, 21), candidate(1, 6)];
    let decision = engine
        .shadow(
            &RequestFacts::default(),
            &PoolFacts::default(),
            &rows,
            &Constraints::default(),
            &mut Scratch::new(),
        )
        .expect("the binding shadows");
    assert_eq!(
        decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .map(|t| t.id),
        Some(rows[1].id)
    );
}

#[test]
fn shadow_sampling_draws_from_the_seed_and_leaves_the_live_draw_alone() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    let profile = document.profiles["random"].clone();
    document.profiles.insert("second".into(), profile);
    document.bindings[0].shadow = Some(spec::ShadowSpec::Profile("second".into()));
    let engine = document.validate().expect("a shadow binding");
    let control = document.validate().expect("the same binding");
    let rows: Vec<_> = (0..23).map(|row| candidate(row, row)).collect();
    let facts = RequestFacts {
        seed: 137,
        ..RequestFacts::default()
    };
    let pool = PoolFacts::default();
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow draw");
    assert!(shadow.trace().drew_first());
    let again = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("the same shadow draw");
    assert_eq!(shadow, again);
    let without = control.decide(&facts, &pool, &rows, &cx, &mut Scratch::new());
    assert_eq!(
        shadow.pick().map(|pick| pick.targets),
        without.pick().map(|pick| pick.targets)
    );
    assert_eq!(
        engine.decide(&facts, &pool, &rows, &cx, &mut scratch),
        without
    );
}

/// A kept key's shadow is decided over the binding's own set, as the
/// live decision is. Over a pool the liveness doubt leaves nobody it
/// goes to the fewest in flight by the lift of that veto, under a
/// sessions rule and under none, and claims no pin.
#[test]
fn shadow_puts_a_kept_keys_doubted_pool_to_the_liveness_tier() {
    for sessions in [None, Some(SESSIONS)] {
        let profile = Profile {
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            ..least_requests_profile()
        };
        let mut engine = with_shadow(profile, sessions);
        engine.places_by_key(1);
        let mut rows = [candidate(0, 13), candidate(1, 3)];
        for row in &mut rows {
            row.health.stalled = true;
        }
        let facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(59),
            session_counted: true,
            now_micros: 71,
            ..RequestFacts::default()
        };
        let decision = engine
            .shadow(
                &facts,
                &PoolFacts::default(),
                &rows,
                &Constraints::default(),
                &mut Scratch::new(),
            )
            .expect("a shadow decision");
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
        assert_eq!(
            decision
                .pick()
                .and_then(|pick| pick.targets.first())
                .map(|t| t.id),
            Some(rows[1].id)
        );
        assert!(engine.sessions().is_empty());
    }
}

/// A twin: the shadow profile holds the live profile's stages under
/// another place. Asked behind the live decision, made through the
/// keeping entry, with that decision's scratch, it decides as the live
/// one does on every path: wherever the shadow's entry leaves the live
/// one's, the two disagree here.
#[test]
fn shadow_twin_decides_as_the_live_profile_on_every_path() {
    let stages = || Profile {
        filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
        ..least_requests_profile()
    };
    let twin = |sessions: Option<SessionRule>, keeps: bool| {
        let mut engine = Engine::new(
            vec![stages(), stages()],
            vec![Binding {
                shadow: Some(1),
                sessions,
                ..single_binding()
            }],
            23,
            Needs::default(),
        );
        if keeps {
            engine.places_by_key(0);
            engine.places_by_key(1);
        }
        engine
    };
    // What a decision says, apart from the profile that made it and
    // the claim a live decision alone makes.
    let said = |decision: &Decision| match decision {
        Decision::Pick(pick) => (
            pick.targets.iter().map(|t| t.id).collect::<Vec<_>>(),
            None,
            pick.trace.lifted,
        ),
        Decision::Shed(shed) => (
            Vec::new(),
            Some((shed.reason, shed.emptied_by)),
            shed.trace.lifted,
        ),
        Decision::Queue(queue) => (Vec::new(), None, queue.trace.lifted),
    };
    let free = [candidate(0, 13), candidate(1, 3)];
    let with = |mark: fn(&mut Candidate)| {
        let mut rows = free;
        rows.iter_mut().for_each(mark);
        rows
    };
    let over = with(|row| row.health.overloaded = true);
    let doubted = with(|row| row.health.stalled = true);
    let down = with(|row| row.health.healthy = false);
    let mut one_over = free;
    one_over[1].health.overloaded = true;
    // Prefill workers: no key is kept over them.
    let mut prefill = with(|row| row.role = Role::Prefill);
    prefill[1].health.overloaded = true;
    // Nor over decode or encode workers: every role but the regular
    // one passes the key on, as the gateway's own placement asks that
    // every candidate is a regular worker (`select_pass`, at main
    // `450084124`, `model_gateway/src/routers/common/placement.rs:1337`).
    let mut decode = with(|row| row.role = Role::Decode);
    decode[1].health.overloaded = true;
    let mut encode = with(|row| row.role = Role::Encode);
    encode[1].health.overloaded = true;
    let plain = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let keyed = RequestFacts {
        session_key: Identity::hashed(67),
        session_counted: true,
        now_micros: 71,
        ..plain
    };
    // Each path: its rule, whether the key is kept, the request, the
    // rows, and whether the key holds a pin where it is decided.
    let paths = [
        ("plain", None, false, &plain, &free, false),
        (
            "plain, every worker over",
            None,
            false,
            &plain,
            &over,
            false,
        ),
        (
            "plain, every worker doubted",
            None,
            false,
            &plain,
            &doubted,
            false,
        ),
        (
            "plain, every worker down",
            None,
            false,
            &plain,
            &down,
            false,
        ),
        (
            "keyed under a rule",
            Some(SESSIONS),
            false,
            &keyed,
            &free,
            false,
        ),
        (
            "keyed under a rule, a pin",
            Some(SESSIONS),
            false,
            &keyed,
            &free,
            true,
        ),
        (
            "keyed under a rule, every worker down",
            Some(SESSIONS),
            false,
            &keyed,
            &down,
            false,
        ),
        ("kept", None, true, &keyed, &one_over, false),
        ("kept, every worker over", None, true, &keyed, &over, false),
        (
            "kept, every worker doubted",
            None,
            true,
            &keyed,
            &doubted,
            false,
        ),
        (
            "kept under a rule",
            Some(SESSIONS),
            true,
            &keyed,
            &one_over,
            false,
        ),
        (
            "kept under a rule, a pin over",
            Some(SESSIONS),
            true,
            &keyed,
            &one_over,
            true,
        ),
        (
            "kept under a rule, every worker over",
            Some(SESSIONS),
            true,
            &keyed,
            &over,
            false,
        ),
        (
            "kept under a rule, every worker doubted",
            Some(SESSIONS),
            true,
            &keyed,
            &doubted,
            false,
        ),
        ("kept, prefill workers", None, true, &keyed, &prefill, false),
        (
            "kept under a rule, prefill workers",
            Some(SESSIONS),
            true,
            &keyed,
            &prefill,
            false,
        ),
        ("kept, decode workers", None, true, &keyed, &decode, false),
        ("kept, encode workers", None, true, &keyed, &encode, false),
    ];
    let cx = &Constraints::default();
    for (path, sessions, keeps, facts, rows, pinned) in paths {
        let pool = PoolFacts::default();
        let mut scratch = Scratch::new();
        let made = || {
            let engine = twin(sessions, keeps);
            if pinned {
                // The key's pin, as an earlier dispatch left it.
                let table = engine.sessions();
                table.lookup(facts.session(), 10, &SESSIONS);
                table.dispatched(facts.session(), SessionVerdict::Pinned, free[1].id, 0);
            }
            engine
        };
        // What the live decision alone leaves of the key.
        let alone = made();
        let _ = alone.decide_keeping(facts, &pool, rows, cx, &mut scratch);
        let engine = made();
        let live = engine.decide_keeping(facts, &pool, rows, cx, &mut scratch);
        let shadow = engine
            .shadow(facts, &pool, rows, cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(said(&shadow), said(&live), "{path}");
        assert_eq!(shadow.trace().profile, 1, "{path}");
        // A worker of another role over its thresholds passes the key
        // on: a kept key would take it.
        if rows[0].role != Role::Single {
            assert_eq!(said(&live).0, [free[0].id], "{path}");
        }
        if !facts.session_key.is_none() {
            assert_eq!(
                engine.sessions().held(facts.session()),
                alone.sessions().held(facts.session()),
                "{path}: the shadow wrote"
            );
        }
    }
}

/// A commit moves the key's pins, so a shadow asked behind it starts
/// from the pins its live decision read. A key no pin held is placed by
/// the shadow's own stages behind the claim's commit: the two disagree
/// where the profiles prefer different rows. The key's next request
/// reads the pin, and the two agree.
#[test]
fn shadow_behind_a_claims_commit_places_the_key_by_its_own_stages() {
    let engine = with_shadow(by_sessions(), Some(SESSIONS));
    let mut busy = candidate(1, 3);
    busy.active_sessions = 17;
    let rows = [candidate(0, 13), busy];
    let facts = keyed_at(61, 71);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert!(matches!(
        live.pick().map(|pick| pick.session_verdict),
        Some(SessionVerdict::Claimed(_))
    ));
    assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[0].id));
    assert_eq!(
        decision::ShadowKind::of(&live, &shadow),
        decision::ShadowKind::OtherTarget
    );
    // The key's next request: the pin stood before its decision.
    let next = keyed_at(61, 89);
    let live = engine.decide_keeping(&next, &pool, &rows, &cx, &mut scratch);
    assert_eq!(
        live.pick().map(|pick| pick.session_verdict),
        Some(SessionVerdict::Pinned)
    );
    assert_eq!(commit(&engine, &next, &live, 0), rows[1].id);
    let shadow = engine
        .shadow(&next, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[1].id));
}

/// A live respill fronts the worker it takes where it is decided, and
/// its commit fronts the worker it landed on. The shadow asked behind
/// it still reads the pin at its cap, and hands the key on by its own
/// stages: a respill of its own, which moves no pin.
#[test]
fn shadow_behind_a_respills_commit_hands_the_key_on_by_its_own_stages() {
    let engine = with_shadow(by_sessions(), Some(SESSIONS));
    let mut rows = [candidate(0, 1), candidate(1, 3), candidate(2, 13)];
    assert_eq!(place_keyed(&engine, &rows, 61, 71).id, rows[0].id);
    // The pin at its cap. The live stages prefer the second row, the
    // shadow's the third.
    rows[0].session_in_flight = 2;
    rows[0].in_flight = 5;
    rows[0].active_sessions = 9;
    rows[1].active_sessions = 17;
    let facts = keyed_at(61, 89);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(
        live.pick().map(|pick| pick.session_verdict),
        Some(SessionVerdict::Respill)
    );
    let moved = Some([rows[1].id, rows[0].id]);
    let pins = || engine.sessions().pins(facts.session());
    assert_eq!(pins(), moved, "ahead of the live commit");
    assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
    assert_eq!(pins(), moved);
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[2].id));
    assert_eq!(
        shadow.pick().map(|pick| pick.session_verdict),
        Some(SessionVerdict::Respill)
    );
    assert_eq!(pins(), moved, "the shadow's respill moves no pin");
}

/// A claim's commit on an alternate withdraws the claim and fronts the
/// alternate. A twin asked behind it still names the live decision's
/// targets, in the plain form and in the form of a kept key.
#[test]
fn shadow_twin_behind_a_commit_on_an_alternate_names_the_live_targets() {
    let stages = || Profile {
        filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
        ..least_requests_profile()
    };
    let targets = |decision: &Decision| {
        decision
            .pick()
            .map(|pick| pick.targets.iter().map(|t| t.id).collect::<Vec<_>>())
    };
    for keeps in [false, true] {
        let mut engine = Engine::new(
            vec![stages(), stages()],
            vec![Binding {
                shadow: Some(1),
                sessions: Some(SESSIONS),
                ..single_binding()
            }],
            23,
            Needs::default(),
        );
        if keeps {
            engine.places_by_key(0);
            engine.places_by_key(1);
        }
        let rows = [candidate(0, 13), candidate(1, 3)];
        let facts = keyed_at(61, 71);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(
            targets(&live),
            Some(vec![rows[1].id, rows[0].id]),
            "kept: {keeps}"
        );
        assert_eq!(commit(&engine, &facts, &live, 1), rows[0].id);
        assert_eq!(
            engine.sessions().pins(facts.session()),
            Some([rows[0].id, WorkerId(0)]),
            "kept: {keeps}"
        );
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(targets(&shadow), targets(&live), "kept: {keeps}");
    }
}

/// The live filters dropped the key's pin, and the live decision
/// claimed another worker. A shadow whose filters keep the pin still
/// follows it behind the claim's commit: the pins the live decision
/// read hold it.
#[test]
fn shadow_behind_a_claims_commit_follows_the_pin_its_own_filters_keep() {
    let live_stages = Profile {
        filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
        ..least_requests_profile()
    };
    let engine = Engine::new(
        vec![live_stages, least_requests_profile()],
        vec![Binding {
            shadow: Some(1),
            sessions: Some(SESSIONS),
            ..single_binding()
        }],
        23,
        Needs::default(),
    );
    let mut rows = [candidate(0, 1), candidate(1, 3), candidate(2, 13)];
    assert_eq!(place_keyed(&engine, &rows, 61, 71).id, rows[0].id);
    rows[0].health.overloaded = true;
    let facts = keyed_at(61, 89);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert!(matches!(
        live.pick().map(|pick| pick.session_verdict),
        Some(SessionVerdict::Claimed(_))
    ));
    assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[0].id));
}

/// The plain entry keeps no reading, under a binding that names a
/// shadow or under none: it is the decision it was. The keeping entry
/// keeps one.
#[test]
fn shadow_reading_is_kept_by_the_keeping_entry_alone() {
    let rows = [candidate(0, 13), candidate(1, 3)];
    let facts = keyed_at(61, 71);
    let key = facts.session().get();
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    for engine in [
        session_engine(SESSIONS),
        with_shadow(by_sessions(), Some(SESSIONS)),
    ] {
        let _ = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
        assert!(!scratch.sessions_found.is_of(key, &facts));
    }
    let engine = with_shadow(by_sessions(), Some(SESSIONS));
    let _ = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert!(scratch.sessions_found.is_of(key, &facts));
}

/// A keeping decision that answers before it reads its key leaves no
/// reading of an earlier one. The same request is decided again, and
/// the live filters keep nobody. Its shadow, whose filters keep the
/// pool, reads the pins as they stand: the worker the first decision's
/// commit pinned, where its own stages would place another.
#[test]
fn shadow_behind_a_decision_that_read_nothing_takes_no_earlier_reading() {
    let live_stages = Profile {
        filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
        ..least_requests_profile()
    };
    let engine = Engine::new(
        vec![live_stages, least_requests_profile()],
        vec![Binding {
            shadow: Some(1),
            sessions: Some(SESSIONS),
            ..single_binding()
        }],
        23,
        Needs::default(),
    );
    let mut rows = [candidate(0, 1), candidate(1, 3)];
    let facts = keyed_at(61, 71);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(commit(&engine, &facts, &live, 0), rows[0].id);
    rows[0].in_flight = 9;
    for row in &mut rows {
        row.health.overloaded = true;
    }
    let _ = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert!(!scratch.sessions_found.is_of(facts.session().get(), &facts));
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[0].id));
}

/// A host that shadows a request decides it through the keeping entry.
/// Here one mixes the two: a keeping decision is committed, and the
/// same request is decided again through the entry that keeps none.
/// The scratch still holds the first decision's reading, and a debug
/// build stops where the shadow would take it.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the shadow takes a reading behind a plain decision")]
fn shadow_behind_a_plain_decision_of_a_kept_request_stops_a_debug_build() {
    let engine = with_shadow(least_requests_profile(), Some(SESSIONS));
    let mut rows = [candidate(0, 1), candidate(1, 3)];
    let facts = keyed_at(61, 71);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(commit(&engine, &facts, &live, 0), rows[0].id);
    rows[0].in_flight = 9;
    let again = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(first_of(&again), Some(rows[0].id));
    let _ = engine.shadow(&facts, &pool, &rows, &cx, &mut scratch);
}

/// The order a host that shadows keeps: each decision of the request
/// through the keeping entry. Behind the second one the shadow starts
/// from what that decision read, the pin the first one's commit left,
/// where its own stages would place another worker.
#[test]
fn shadow_behind_a_second_keeping_decision_starts_from_its_reading() {
    let engine = with_shadow(least_requests_profile(), Some(SESSIONS));
    let mut rows = [candidate(0, 1), candidate(1, 3)];
    let facts = keyed_at(61, 71);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(commit(&engine, &facts, &live, 0), rows[0].id);
    rows[0].in_flight = 9;
    let again = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(first_of(&again), Some(rows[0].id));
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[0].id));
}

/// The reading holds both pins the live decision read. The key's first
/// pin is unhealthy, the live decision fails over to the second, and
/// the shadow fails over to it too: with the second pin lost it would
/// place the key on an unpinned worker by its own stages.
#[test]
fn shadow_fails_over_to_the_second_pin_the_live_decision_read() {
    let engine = with_shadow(least_requests_profile(), Some(SESSIONS));
    let mut rows = [candidate(0, 13), candidate(1, 3), candidate(2, 1)];
    let facts = keyed_at(61, 71);
    let key = facts.session();
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    engine.sessions().lookup(key, facts.now_micros, &SESSIONS);
    for row in [&rows[1], &rows[0]] {
        engine
            .sessions()
            .dispatched(key, SessionVerdict::Respill, row.id, 0);
    }
    assert_eq!(engine.sessions().pins(key), Some([rows[0].id, rows[1].id]));
    rows[0].health.healthy = false;
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(first_of(&live), Some(rows[1].id));
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[1].id));
}

/// A reading is of one request: its key, its clock and its seed.
/// Another request of the same key at the same clock takes none of
/// it. Its shadow reads the pins as they stand, the worker the first
/// request's commit pinned, where the reading held no pin.
#[test]
fn shadow_of_a_request_of_another_seed_takes_no_reading() {
    let engine = with_shadow(by_sessions(), Some(SESSIONS));
    let mut busy = candidate(1, 3);
    busy.active_sessions = 17;
    let rows = [candidate(0, 13), busy];
    let facts = keyed_at(61, 71);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
    let next = RequestFacts { seed: 1, ..facts };
    let shadow = engine
        .shadow(&next, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[1].id));
}

/// The reading holds the key's pins as the table held them, whether or
/// not the live filters keep their workers. Here they drop the pin's
/// worker and the live decision claims another. The shadow, whose
/// filters keep that worker, follows the pin the live decision read:
/// the claim's commit has moved the pins since.
#[test]
fn shadow_follows_a_pin_the_live_filters_left_out() {
    let live_stages = Profile {
        filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
        ..least_requests_profile()
    };
    let engine = Engine::new(
        vec![live_stages, least_requests_profile()],
        vec![Binding {
            shadow: Some(1),
            sessions: Some(SESSIONS),
            ..single_binding()
        }],
        23,
        Needs::default(),
    );
    let mut rows = [candidate(0, 13), candidate(1, 3)];
    rows[0].health.overloaded = true;
    let facts = keyed_at(61, 71);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let table = engine.sessions();
    table.lookup(facts.session(), 10, &SESSIONS);
    table.dispatched(facts.session(), SessionVerdict::Pinned, rows[0].id, 0);
    let mut scratch = Scratch::new();
    let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
    assert_eq!(table.pins(facts.session()), Some([rows[1].id, rows[0].id]));
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(first_of(&shadow), Some(rows[0].id));
}

/// The best survivor, picked once another thread has had its turn at
/// the table: the stages place a request between its read of the key
/// and its claim, and one thread has no seam there.
#[derive(Debug)]
struct Turned {
    go: mpsc::Sender<()>,
    done: Mutex<mpsc::Receiver<()>>,
    once: Once,
}

impl Turned {
    /// The picker, with the two ends the other thread holds: it waits
    /// on the first for the pick, has its turn at the table, and says
    /// so on the second.
    fn new() -> (Self, mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (go, went) = mpsc::channel();
        let (did, done) = mpsc::channel();
        let picker = Self {
            go,
            done: Mutex::new(done),
            once: Once::new(),
        };
        (picker, went, did)
    }
}

impl Lifecycle for Turned {}

impl Picker for Turned {
    fn name(&self) -> &'static str {
        "turned"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        self.once.call_once(|| {
            let _ = self.go.send(());
            if let Ok(done) = self.done.lock() {
                let _ = done.recv_timeout(Duration::from_secs(10));
            }
        });
        Best.pick(facts, candidates, kept, scores, band, out);
    }
}

/// Another request's respill turns the key's pins between a decision's
/// read and its claim. The claim is not taken and no pin came in, so
/// the table answers with the decision's own pick. That pick is no
/// pin of the key. A shadow of another profile reads the pins the
/// claim found against its own pick and keeps its own placement: it
/// counts the other target it chose, and no false agreement.
#[test]
fn shadow_of_another_profile_keeps_its_pick_where_the_claim_found_no_new_pin() {
    let (picker, went, did) = Turned::new();
    let live = Profile {
        picker: Box::new(picker),
        ..least_requests_profile()
    };
    let other = Profile {
        name: "first".into(),
        picker: Box::new(FirstKept),
        ..least_requests_profile()
    };
    let engine = Engine::new(
        vec![live, other],
        vec![Binding {
            shadow: Some(1),
            sessions: Some(SESSIONS),
            ..single_binding()
        }],
        23,
        Needs::default(),
    );
    let mut rows = [
        candidate(0, 0),
        candidate(1, 0),
        candidate(2, 1),
        candidate(3, 0),
    ];
    rows[0].health.healthy = false;
    rows[1].health.healthy = false;
    let facts = keyed_at(61, 71);
    let pool = PoolFacts::default();
    let key = facts.session();
    let table = engine.sessions();
    table.lookup(key, 10, &SESSIONS);
    table.dispatched(key, SessionVerdict::Pinned, rows[0].id, 0);
    table.dispatched(key, SessionVerdict::Pinned, rows[1].id, 0);
    assert_eq!(table.pins(key), Some([rows[1].id, rows[0].id]));
    let turned = rows[0].id;
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    let live = std::thread::scope(|scope| {
        scope.spawn(move || {
            // Another request's respill, once the live decision has
            // read the key: both pins stay, turned.
            if went.recv_timeout(Duration::from_secs(10)).is_ok() {
                table.dispatched(key, SessionVerdict::Respill, turned, 0);
            }
            let _ = did.send(());
        });
        engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch)
    });
    // The claim was not taken and nothing was recorded: the pins stand
    // as the respill left them.
    assert_eq!(table.pins(key), Some([rows[0].id, rows[1].id]));
    assert_eq!(first_of(&live), Some(rows[3].id));
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow decision");
    assert_eq!(
        decision::ShadowKind::of(&live, &shadow),
        decision::ShadowKind::OtherTarget
    );
    assert_eq!(first_of(&shadow), Some(rows[2].id));
}

/// Another request's pin comes in between a decision's read of the key
/// and its claim. The key holds a pin already, on a worker that is not
/// ready, and the newcomer joins behind it. The claim is not taken,
/// the table answers with the newcomer, and the live decision follows
/// it. A shadow of another profile, whose own stages place the key
/// elsewhere, reads the pins the claim found against the pins it
/// started from and follows the newcomer too. It takes them at the
/// claim, not at its read. In the kept form the newcomer is over its
/// thresholds: read first, the pins would hold no worker the shadow's
/// standing pass keeps, and it would keep its own placement.
#[test]
fn shadow_follows_the_pin_that_came_in_where_the_claim_was_not_taken() {
    let rule = SessionRule {
        recovery: Recovery::Original,
        ..SESSIONS
    };
    for kept in [false, true] {
        let (picker, went, did) = Turned::new();
        let live = Profile {
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            picker: Box::new(picker),
            ..least_requests_profile()
        };
        let other = Profile {
            name: "first".into(),
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            picker: Box::new(FirstKept),
            ..least_requests_profile()
        };
        let mut engine = Engine::new(
            vec![live, other],
            vec![Binding {
                shadow: Some(1),
                sessions: Some(rule),
                ..single_binding()
            }],
            23,
            Needs::default(),
        );
        if kept {
            engine.places_by_key(0);
            engine.places_by_key(1);
        }
        // The shadow's stages place the key on the first row and the
        // live ones on the third. The newcomer is the row between
        // them, and the key's pin is the last row, which is not ready.
        let mut rows = [
            candidate(0, 5),
            candidate(1, 3),
            candidate(2, 0),
            candidate(3, 0),
        ];
        rows[1].health.overloaded = kept;
        rows[3].health.healthy = false;
        let (newcomer, pinned) = (rows[1].id, rows[3].id);
        let facts = keyed_at(61, 71);
        let pool = PoolFacts::default();
        let key = facts.session();
        let table = engine.sessions();
        table.lookup(key, 10, &rule);
        table.dispatched(key, SessionVerdict::Pinned, pinned, 0);
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let live = std::thread::scope(|scope| {
            scope.spawn(move || {
                // Another request's dispatch, once the live decision
                // has read the key: the newcomer joins behind the pin.
                if went.recv_timeout(Duration::from_secs(10)).is_ok() {
                    table.dispatched(key, SessionVerdict::Pinned, newcomer, 0);
                }
                let _ = did.send(());
            });
            engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch)
        });
        // The claim was not taken and nothing was recorded.
        assert_eq!(table.pins(key), Some([pinned, newcomer]), "kept: {kept}");
        assert_eq!(first_of(&live), Some(newcomer), "kept: {kept}");
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(newcomer), "kept: {kept}");
        assert_eq!(
            decision::ShadowKind::of(&live, &shadow),
            decision::ShadowKind::Agreed,
            "kept: {kept}"
        );
    }
}

#[test]
fn shadow_uses_session_pins_without_claiming_touching_or_sweeping() {
    let rule = SessionRule {
        ttl_micros: Some(41),
        sweep_micros: 11,
        ..SESSIONS
    };
    let engine = with_shadow(by_sessions(), Some(rule));
    let control = with_shadow(by_sessions(), Some(rule));
    let mut busy = candidate(1, 3);
    busy.active_sessions = 17;
    let rows = [candidate(0, 13), busy];
    let mut facts = RequestFacts {
        model: "m",
        session_key: Identity::hashed(59),
        now_micros: 71,
        ..RequestFacts::default()
    };
    let key = facts.session();
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let first = |decision: &Decision| {
        decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .map(|t| t.id)
    };
    let fresh = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a shadow placement");
    assert_eq!(first(&fresh), Some(rows[0].id));
    assert_eq!(
        fresh.pick().map(|pick| pick.session_verdict),
        Some(SessionVerdict::None)
    );
    assert!(engine.sessions().is_empty());
    let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
    assert_eq!(first(&live), Some(rows[1].id));
    assert_eq!(
        live,
        control.decide(&facts, &pool, &rows, &cx, &mut Scratch::new())
    );
    let pinned = control.sessions().held(key);
    assert!(pinned.is_some());
    assert_eq!(engine.sessions().held(key), pinned);

    facts.now_micros += 17;
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a pinned shadow");
    assert_eq!(first(&shadow), first(&live));
    assert_eq!(
        shadow.pick().map(|pick| pick.session_verdict),
        Some(SessionVerdict::Pinned)
    );
    assert_eq!(engine.sessions().held(key), pinned);

    // Idle until 112: the rule's next sweep is at 121.
    facts.now_micros += 33;
    let expired = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("an expired pin leaves a fresh placement");
    assert_eq!(first(&expired), Some(rows[0].id));
    assert_eq!(engine.sessions().held(key), pinned);

    let untimed = with_shadow(by_sessions(), Some(SESSIONS));
    let live = untimed.decide(&facts, &pool, &rows, &cx, &mut scratch);
    facts.now_micros += 71;
    let shadow = untimed
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("a pin with no expiry");
    assert_eq!(first(&shadow), first(&live));
}

#[test]
fn shadow_can_keep_a_key_over_the_overload_veto_and_writes_no_pin() {
    for sessions in [None, Some(SESSIONS)] {
        let profile = Profile {
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            ..least_requests_profile()
        };
        let mut engine = with_shadow(profile, sessions);
        engine.places_by_key(1);
        let mut rows = [candidate(0, 3), candidate(1, 13)];
        rows[0].health.overloaded = true;
        let facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(59),
            session_counted: true,
            now_micros: 71,
            ..RequestFacts::default()
        };
        let decision = engine
            .shadow(
                &facts,
                &PoolFacts::default(),
                &rows,
                &Constraints::default(),
                &mut Scratch::new(),
            )
            .expect("a shadow placement");
        assert_eq!(
            decision
                .pick()
                .and_then(|pick| pick.targets.first())
                .map(|t| t.id),
            Some(rows[0].id)
        );
        assert!(engine.sessions().is_empty());
    }
}

#[test]
fn shadow_stages_count_live_dispatches_but_a_shadow_pick_sends_nothing() {
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Debug)]
    struct Dispatches(Arc<AtomicU32>);

    impl Lifecycle for Dispatches {
        fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl Scorer for Dispatches {
        fn name(&self) -> &'static str {
            "dispatch-count"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            rows: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            let count = self.0.load(Ordering::Relaxed);
            for (score, &i) in out.iter_mut().zip(kept) {
                *score = if rows[i as usize].position == count {
                    1.0
                } else {
                    0.0
                };
            }
            TieBand::NORMALIZED
        }
    }

    let count = Arc::new(AtomicU32::new(0));
    let engine = with_shadow(
        Profile {
            scorers: vec![WeightedScorer {
                scorer: Box::new(Dispatches(Arc::clone(&count))),
                weight: 1.0,
            }],
            ..least_requests_profile()
        },
        None,
    );
    let rows = [candidate(0, 3), candidate(1, 13)];
    let (facts, pool, cx) = (
        RequestFacts::default(),
        PoolFacts::default(),
        Constraints::default(),
    );
    let mut scratch = Scratch::new();
    let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
    let target = live
        .pick()
        .and_then(|pick| pick.targets.first())
        .expect("a live target");
    assert_eq!(target.id, rows[0].id);
    engine.on_dispatched(&Lease::default(), target, Side::Single, 0);
    let shadow = engine
        .shadow(&facts, &pool, &rows, &cx, &mut scratch)
        .expect("the binding shadows");
    assert_eq!(
        shadow
            .pick()
            .and_then(|pick| pick.targets.first())
            .map(|t| t.id),
        Some(rows[1].id)
    );
    assert_eq!(count.load(Ordering::Relaxed), 1);
}

#[test]
fn shadow_flags_compose_with_sampling_and_plain_bindings_have_no_shadow() {
    let mut plain = engine();
    let facts = RequestFacts::default();
    assert_eq!(plain.seat_flags(), 0);
    plain.set_explain_sample(17);
    assert_eq!(plain.seat_flags(), 2);
    assert!(plain
        .shadow(
            &facts,
            &PoolFacts::default(),
            &[],
            &Constraints::default(),
            &mut Scratch::new(),
        )
        .is_none());
    let mut shadowed = with_shadow(by_sessions(), None);
    assert_eq!(shadowed.seat_flags(), 4);
    shadowed.set_explain_sample(17);
    assert_eq!(shadowed.seat_flags(), 6);
    shadowed.set_explain_sample(0);
    assert_eq!(shadowed.seat_flags(), 4);
}

#[test]
fn shadow_respects_binding_order_and_separate_profile_state() {
    let fallback = Binding {
        shadow: Some(1),
        ..single_binding()
    };
    let first = Binding {
        model: "first".into(),
        ..single_binding()
    };
    let build = |first, fallback| {
        Engine::new(
            vec![least_requests_profile(), by_sessions()],
            vec![first, fallback],
            23,
            Needs::default(),
        )
    };
    let run = |engine: &Engine, model| {
        engine.shadow(
            &RequestFacts {
                model,
                ..RequestFacts::default()
            },
            &PoolFacts::default(),
            &[],
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    let engine = build(first.clone(), fallback.clone());
    assert!(run(&engine, "first").is_none());
    assert!(run(&engine, "second").is_some());
    let reused = build(
        Binding {
            profile: 1,
            ..first
        },
        fallback,
    );
    assert!(run(&reused, "second").is_none());
}
