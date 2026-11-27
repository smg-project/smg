//! The rule for one list whose every worker is over its thresholds: the
//! steer, the binding's word that sheds, and what the rule leaves alone.

use super::*;

/// Every preset whose stages this build carries steers a pool that is
/// over its thresholds on every worker to the fewest requests in flight,
/// whatever its own stages would pick, and names the next fewest as
/// alternates.
#[test]
fn a_pool_over_on_every_worker_goes_to_the_fewest_in_flight_under_every_preset() {
    let pool = [over(0, 3), over(1, 1), over(2, 2)];
    for preset in [
        "random",
        "round_robin",
        "passthrough",
        "power_of_two",
        "least_load",
    ] {
        let engine = engine_of(preset, false);
        for seed in 1..=8 {
            let decision = decide(&engine, &facts(seed), &pool);
            assert_eq!(order(&decision), vec![1, 2, 0], "{preset}, seed {seed}");
            let pick = decision.pick().expect("a pick");
            assert_eq!(pick.trace.survivors.get(), Some(3), "{preset}");
            assert_eq!(pick.session_verdict, SessionVerdict::None, "{preset}");
        }
    }
}

/// Under the binding's word the same pool is shed as saturated, by the
/// overload filter's name.
#[test]
fn the_same_pool_is_shed_as_saturated_under_the_bindings_word() {
    let pool = [over(0, 3), over(1, 1), over(2, 2)];
    for preset in ["random", "round_robin", "least_load", "manual"] {
        let decision = decide(&engine_of(preset, true), &facts(1), &pool);
        assert_eq!(
            shed_of(&decision),
            (ShedReason::Saturated, "not-overloaded"),
            "{preset}"
        );
    }
}

/// A hard filter listed behind the overload filter does not hide an
/// all-vetoed pool: the one worker with a free slot is over its
/// thresholds and the free worker has no slot, so the later filter empties
/// the pass, and the request still goes to the usable worker. The answer
/// is the same with the two filters listed the other way round. Where no
/// worker has a slot, lifting the veto gains nothing and the shed names
/// the filter that says why, also when the veto itself emptied the first
/// pass. A binding that sheds answers at the stage that emptied its
/// pass, as before.
#[test]
fn a_hard_filter_behind_the_veto_does_not_hide_an_all_vetoed_pool() {
    let mut pool = [over(0, 4), over(1, 0)];
    pool[0].has_capacity = true;
    pool[1].health.overloaded = false;
    let behind = ["healthy", "not-overloaded", "has-capacity", "not-tried"];
    let ahead = ["healthy", "has-capacity", "not-overloaded", "not-tried"];
    for filters in [behind, ahead] {
        let decision = decide(&engine_filtering(&filters, false), &facts(1), &pool);
        assert_eq!(order(&decision), vec![0], "{filters:?}");
    }
    assert_eq!(
        shed_of(&decide(&engine_filtering(&behind, true), &facts(1), &pool)),
        (ShedReason::NoEndpoints, "has-capacity")
    );
    assert_eq!(
        shed_of(&decide(&engine_filtering(&ahead, true), &facts(1), &pool)),
        (ShedReason::Saturated, "not-overloaded")
    );
    pool[0].has_capacity = false;
    for filters in [behind, ahead] {
        let decision = decide(&engine_filtering(&filters, false), &facts(1), &pool);
        assert_eq!(
            shed_of(&decision),
            (ShedReason::NoEndpoints, "has-capacity"),
            "{filters:?}"
        );
    }
    // Every worker over its thresholds and none with a slot: the veto
    // empties the first pass, the later filter empties the second, and
    // the shed names that filter, not the veto.
    pool[1].health.overloaded = true;
    assert_eq!(
        shed_of(&decide(&engine_filtering(&behind, false), &facts(1), &pool)),
        (ShedReason::NoEndpoints, "has-capacity")
    );
}

/// The law the example above is one instance of: the overload veto is the
/// one filter that steps aside, so the pool is what the hard filters keep
/// less the vetoed, or what the hard filters keep when that leaves
/// nobody. Under every order of three hard filters (one of them the
/// veto) the steer is the same two workers, fewest in flight first; and
/// with one usable worker under its thresholds every order picks it.
#[test]
fn the_pick_is_the_same_under_every_order_of_the_hard_filters() {
    let orders = [
        ["healthy", "not-overloaded", "has-capacity"],
        ["healthy", "has-capacity", "not-overloaded"],
        ["not-overloaded", "healthy", "has-capacity"],
        ["not-overloaded", "has-capacity", "healthy"],
        ["has-capacity", "healthy", "not-overloaded"],
        ["has-capacity", "not-overloaded", "healthy"],
    ];
    // Usable and over its thresholds: rows 0 and 3. Row 1 is free with no
    // slot; row 2 is free with a slot and draining.
    let mut pool = [over(0, 4), over(1, 0), over(2, 0), over(3, 6)];
    pool[0].has_capacity = true;
    pool[1].health.overloaded = false;
    pool[2].health.overloaded = false;
    pool[2].has_capacity = true;
    pool[2].health.draining = true;
    pool[3].has_capacity = true;
    for hard in orders {
        let filters = [hard[0], hard[1], hard[2], "not-tried"];
        let engine = engine_filtering(&filters, false);
        for seed in 1..=4 {
            let decision = decide(&engine, &facts(seed), &pool);
            assert_eq!(order(&decision), vec![0, 3], "{hard:?}, seed {seed}");
        }
    }
    // Row 1 gains a slot: one usable worker under its thresholds.
    pool[1].has_capacity = true;
    for hard in orders {
        let filters = [hard[0], hard[1], hard[2], "not-tried"];
        let decision = decide(&engine_filtering(&filters, false), &facts(1), &pool);
        assert_eq!(order(&decision), vec![1], "{hard:?}");
    }
}

/// Equal counts are one tie, drawn by the decision's seed over the tied
/// rows alone: a row that holds more is never drawn.
#[test]
fn a_tie_is_drawn_by_the_seed_over_the_tied_rows_alone() {
    let engine = engine_of("round_robin", false);
    let pool = [over(0, 2), over(1, 2), over(2, 5)];
    let mut firsts = [0u32; 3];
    for seed in 1..=600 {
        let first = order(&decide(&engine, &facts(seed), &pool))[0];
        firsts[first as usize] += 1;
    }
    assert_eq!(firsts[2], 0, "{firsts:?}");
    assert!(firsts[0] > 200 && firsts[1] > 200, "{firsts:?}");
}

/// A worker that is down, draining or behind an open breaker is not
/// usable: it is never the steer, however few requests it holds. A
/// doubted worker is the last to be taken, once no worker over its
/// thresholds alone is left.
#[test]
fn a_worker_another_veto_refuses_is_never_the_steer() {
    let engine = engine_of("least_load", false);
    let mut pool = [over(0, 0), over(1, 0), over(2, 0), over(3, 0), over(4, 7)];
    pool[0].health.healthy = false;
    pool[1].health.draining = true;
    pool[2].health.stalled = true;
    pool[3].health.breaker_open = true;
    assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![4]);
    // With no worker over its thresholds alone the doubted one is taken.
    pool[4].health.healthy = false;
    assert_eq!(order(&decide(&engine, &facts(1), &pool)), vec![2]);
    // With no usable row left the stage that refused them names the shed.
    pool[2].health.healthy = false;
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &pool)),
        (ShedReason::NoEndpoints, "healthy")
    );
}

/// A worker under its thresholds takes the request by the profile's own
/// stages: the rule runs only when the overload filter leaves nobody.
#[test]
fn a_worker_under_its_thresholds_keeps_the_profiles_own_pick() {
    let engine = engine_of("passthrough", false);
    let mut pool = [over(0, 0), over(1, 9), over(2, 0)];
    pool[1].health.overloaded = false;
    assert_eq!(order(&decide(&engine, &facts(1), &pool))[0], 1);
}

/// The retry's tried list does not narrow the steer: a worker an earlier
/// attempt tried is taken again when it holds the fewest requests.
#[test]
fn a_tried_worker_is_taken_again_when_it_holds_the_fewest() {
    let engine = engine_of("round_robin", false);
    let pool = [over(0, 3), over(1, 1), over(2, 2)];
    let tried = [WorkerId(101)];
    let retry = RequestFacts {
        tried: &tried,
        attempt: 1,
        ..facts(1)
    };
    assert_eq!(order(&decide(&engine, &retry, &pool)), vec![1, 2, 0]);
}

/// The host's list of workers it just failed to reach stays in the
/// lifted pass: the steer passes over such a worker while another usable
/// one stands, whatever it holds, and takes it as the last resort.
#[test]
fn a_worker_the_host_failed_to_reach_is_the_steers_last_resort() {
    let engine = engine_of("round_robin", false);
    let pool = [over(0, 3), over(1, 1), over(2, 2)];
    let steer = |unreachable: &[WorkerId]| {
        let cx = Constraints {
            excluded: unreachable,
            ..Constraints::default()
        };
        let decision = engine.decide(
            &facts(1),
            &PoolFacts::default(),
            &pool,
            &cx,
            &mut Scratch::new(),
        );
        order(&decision)
    };
    assert_eq!(steer(&[WorkerId(101)]), vec![2, 0]);
    assert_eq!(steer(&[WorkerId(101), WorkerId(102)]), vec![0]);
    let every = [WorkerId(100), WorkerId(101), WorkerId(102)];
    assert_eq!(steer(&every), vec![1, 2, 0]);
}

/// Under every preset a retry is steered past the workers its host
/// failed to reach: three attempts over a pool over on every worker
/// take its three workers by their requests in flight, and each pick
/// says the lifted veto.
#[test]
fn a_retry_leaves_the_workers_the_host_failed_to_reach() {
    let pool = [over(0, 3), over(1, 1), over(2, 2)];
    let mut read = Vec::new();
    for preset in [
        "round_robin",
        "least_load",
        "random",
        "power_of_two",
        "cache_aware",
        "consistent_hashing",
        "passthrough",
    ] {
        let engine = engine_of(preset, false);
        let mut failed: Vec<WorkerId> = Vec::new();
        let mut firsts = Vec::new();
        for attempt in 0..3u64 {
            let cx = Constraints {
                excluded: &failed,
                ..Constraints::default()
            };
            let retry = RequestFacts {
                tried: &failed,
                ..facts(1 + attempt)
            };
            let decision = engine.decide(
                &retry,
                &PoolFacts::default(),
                &pool,
                &cx,
                &mut Scratch::new(),
            );
            let first = order(&decision)[0];
            assert_eq!(
                decision.trace().lifted,
                LiftedVeto::Overload,
                "{preset} attempt {attempt}"
            );
            firsts.push(first);
            failed.push(WorkerId(100 + u64::from(first)));
        }
        read.push((preset, firsts));
    }
    assert!(
        read.iter().all(|(_, firsts)| firsts == &[1, 2, 0]),
        "{read:?}"
    );
}

/// A steered pick says the veto the rule lifted, the mark a host counts
/// such picks from. A pick the profile's own stages made says
/// none, and so does the shed of a binding that sheds.
#[test]
fn a_steered_pick_says_the_veto_it_lifted() {
    let engine = engine_of("round_robin", false);
    let steered = decide(&engine, &facts(1), &[over(0, 3), over(1, 1)]);
    assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
    let mut under = over(1, 1);
    under.health.overloaded = false;
    let kept = decide(&engine, &facts(1), &[over(0, 3), under]);
    assert_eq!(kept.trace().lifted, LiftedVeto::None);
    let shedding = engine_of("round_robin", true);
    let shed = decide(&shedding, &facts(1), &[over(0, 3), over(1, 1)]);
    assert_eq!(shed.trace().lifted, LiftedVeto::None);
}

/// A binding the engine does not hold cannot be told to shed: the index
/// is the validator's own, and a debug build refuses any other.
#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "is not built"))]
fn a_binding_the_engine_does_not_hold_is_refused_in_a_debug_build() {
    let mut engine = engine_of("round_robin", false);
    engine.shed_when_all_vetoed(7);
}

/// A host's lift puts the decision outside the veto: the filter keeps
/// every row and the profile's own stages pick.
#[test]
fn a_hosts_lift_keeps_the_profiles_own_pick() {
    let engine = engine_of("passthrough", false);
    let pool = [over(0, 3), over(1, 1), over(2, 2)];
    let lifted = Constraints {
        overload_lifted: true,
        ..Constraints::default()
    };
    let decision = engine.decide(
        &facts(1),
        &PoolFacts::default(),
        &pool,
        &lifted,
        &mut Scratch::new(),
    );
    assert_eq!(order(&decision)[0], 0, "the first survivor, as listed");
}

/// A pool a picker draws from ends in the same rule. The rows the draw
/// names are refused as every other row is, the whole pass keeps nobody
/// either, and the binding's word answers: the fewest in flight, or the
/// shed.
#[test]
fn a_pool_a_picker_draws_from_ends_in_the_same_rule() {
    let rows = u32::try_from(DRAW_FROM).expect("a small count") + 4;
    // The fewest requests in flight stand on the last row.
    let pool: Vec<Candidate> = (0..rows).map(|at| over(at, rows - at)).collect();
    for preset in [
        "random",
        "round_robin",
        "passthrough",
        "power_of_two",
        "least_load",
    ] {
        let steering = engine_of(preset, false);
        let shedding = engine_of(preset, true);
        for seed in 1..=8 {
            let steered = decide(&steering, &facts(seed), &pool);
            assert_eq!(
                order(&steered).first(),
                Some(&(rows - 1)),
                "{preset}, seed {seed}"
            );
            let pick = steered.pick().expect("a pick");
            assert_eq!(pick.trace.survivors.get(), Some(rows), "{preset}");
            assert_eq!(
                shed_of(&decide(&shedding, &facts(seed), &pool)),
                (ShedReason::Saturated, "not-overloaded"),
                "{preset}, seed {seed}"
            );
        }
    }
}

/// A request that names rows ends in the same rule. Under a picker the
/// rows named join, the draw is asked for such a request: the rows it
/// names are refused as every other row is, the whole pass keeps nobody
/// either, and the binding's word answers: the fewest in flight, which
/// no name stands on, or the shed.
#[test]
fn a_request_that_names_rows_ends_in_the_same_rule() {
    let rows = u32::try_from(DRAW_FROM).expect("a small count") + 4;
    // The fewest requests in flight stand on the last row; the host
    // names two others.
    let pool: Vec<Candidate> = (0..rows).map(|at| over(at, rows - at)).collect();
    static HASHES: [u64; 16] = [0; 16];
    let holders = [
        AffinityHolder {
            row: 3,
            overlap_blocks: 12,
            predicted_blocks: 0,
        },
        AffinityHolder {
            row: 7,
            overlap_blocks: 8,
            predicted_blocks: 0,
        },
    ];
    let steering = engine_of("cache_aware", false);
    let shedding = engine_of("cache_aware", true);
    for seed in 1..=8 {
        let naming = RequestFacts {
            block_hashes: &HASHES,
            prompt_tokens: 256,
            named: Some(AffinityRows {
                holders: &holders,
                ..AffinityRows::default()
            }),
            ..facts(seed)
        };
        let steered = decide(&steering, &naming, &pool);
        assert_eq!(order(&steered).first(), Some(&(rows - 1)), "seed {seed}");
        let pick = steered.pick().expect("a pick");
        assert_eq!(pick.trace.survivors.get(), Some(rows), "seed {seed}");
        assert_eq!(
            shed_of(&decide(&shedding, &naming, &pool)),
            (ShedReason::Saturated, "not-overloaded"),
            "seed {seed}"
        );
    }
}

/// The word is the binding's and not the profile's: two bindings that
/// share one profile answer each by its own, over a pool the whole pass
/// decides and over one a picker draws from. The binding that sheds
/// stands first and then second, so the place the validator names for
/// it is read.
#[test]
fn two_bindings_of_one_profile_answer_each_by_its_own_word() {
    let sheds = || {
        Some(OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        })
    };
    let drawn = u32::try_from(DRAW_FROM).expect("a small count") + 4;
    for shedding_first in [true, false] {
        let mut spec = PolicySpec::preset("random").expect("the preset's form");
        let wildcard = spec.bindings.first_mut().expect("the preset's binding");
        let mut named = wildcard.clone();
        named.matcher.model = "named".into();
        if shedding_first {
            named.overload = sheds();
        } else {
            wildcard.overload = sheds();
        }
        spec.bindings.insert(0, named);
        let engine = spec.validate().expect("the document builds");
        let (steers, strict) = if shedding_first {
            ("m", "named")
        } else {
            ("named", "m")
        };
        for rows in [3, drawn] {
            // The fewest requests in flight stand on the last row.
            let pool: Vec<Candidate> = (0..rows).map(|at| over(at, rows - at)).collect();
            let steered = RequestFacts {
                model: steers,
                ..facts(1)
            };
            assert_eq!(
                order(&decide(&engine, &steered, &pool)).first(),
                Some(&(rows - 1)),
                "{rows} rows, the shedding binding first: {shedding_first}"
            );
            let shed = RequestFacts {
                model: strict,
                ..facts(1)
            };
            assert_eq!(
                shed_of(&decide(&engine, &shed, &pool)),
                (ShedReason::Saturated, "not-overloaded"),
                "{rows} rows, the shedding binding first: {shedding_first}"
            );
        }
    }
}

/// A shed the veto had no part in is not filtered a second time: a pool
/// emptied before the veto ran, a decision whose host lifted the veto
/// itself and a pool with no row over its thresholds cost the one pass
/// they cost without the rule. A pool with one row over its thresholds
/// is filtered again, and sheds at the stage behind the veto.
#[test]
fn a_shed_the_veto_had_no_part_in_costs_one_pass() {
    let counted = |veto_first: bool| counted(Box::new(NotOverloaded), veto_first);
    let pool = [over(0, 1), over(1, 2)];
    // Listed ahead of the veto: the veto never ran.
    let (engine, asked) = counted(false);
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &pool)),
        (ShedReason::NoEndpoints, "refuses")
    );
    assert_eq!(asked.load(Ordering::Relaxed), pool.len());
    // Listed behind it, under a host's lift: the veto was off already.
    let (engine, asked) = counted(true);
    let lifted = Constraints {
        overload_lifted: true,
        ..Constraints::default()
    };
    let decision = engine.decide(
        &facts(1),
        &PoolFacts::default(),
        &pool,
        &lifted,
        &mut Scratch::new(),
    );
    assert_eq!(shed_of(&decision), (ShedReason::NoEndpoints, "refuses"));
    assert_eq!(asked.load(Ordering::Relaxed), pool.len());
    // Listed behind it, with nobody over its thresholds: the veto ran
    // and refused no row.
    let (engine, asked) = counted(true);
    let mut under = pool;
    for worker in &mut under {
        worker.health.overloaded = false;
    }
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &under)),
        (ShedReason::NoEndpoints, "refuses")
    );
    assert_eq!(asked.load(Ordering::Relaxed), under.len());
    // One row over its thresholds: the veto refused it, so the lifted
    // pass asks the filter behind the veto about both rows.
    let (engine, asked) = counted(true);
    let mut mixed = under;
    mixed[0].health.overloaded = true;
    assert_eq!(
        shed_of(&decide(&engine, &facts(1), &mixed)),
        (ShedReason::NoEndpoints, "refuses")
    );
    assert_eq!(asked.load(Ordering::Relaxed), 1 + mixed.len());
}

/// The by-assignment path (a picker that leaves the placement to its
/// rule, a request with no key) is steered as the plain path is.
#[test]
fn a_request_placed_by_assignment_is_steered_too() {
    let pool = [over(0, 3), over(1, 1), over(2, 2)];
    let decision = decide(&engine_of("manual", false), &facts(1), &pool);
    assert_eq!(order(&decision), vec![1, 2, 0]);
}

/// The keyed path: a key whose pool is over on every worker goes to the
/// fewest in flight and the steer pins nothing, so the key's pin stands
/// for the decision after the pool recovers.
#[test]
fn a_keyed_request_is_steered_and_the_steer_pins_nothing() {
    let engine = engine_of("manual", false);
    let place = |pool: &[Candidate], now: u64| {
        let keyed = RequestFacts {
            session_key: Identity::plain("k1"),
            now_micros: now,
            ..facts(now)
        };
        let decision = decide(&engine, &keyed, pool);
        let pick = *decision.pick().expect("a pick");
        let target = *pick.targets.first().expect("a target");
        let lease = Lease {
            issued_micros: now,
            session: keyed.session(),
            session_verdict: pick.session_verdict,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
        (target.position, pick.session_verdict)
    };
    let mut pool = [over(0, 1), over(1, 2), over(2, 3)];
    for c in &mut pool {
        c.health.overloaded = false;
    }
    let (home, _) = place(&pool, 1_000);
    for c in &mut pool {
        c.health.overloaded = true;
    }
    pool[home as usize].in_flight = 9;
    let (steered, verdict) = place(&pool, 2_000);
    assert_ne!(steered, home, "the fewest in flight, not the pin");
    assert_eq!(verdict, SessionVerdict::None);
    for c in &mut pool {
        c.health.overloaded = false;
    }
    assert_eq!(place(&pool, 3_000).0, home, "the pin stands");
}

/// An encode item of the `epd` plan is steered behind the plan's own
/// rules: an encoder the pair cannot use is not usable.
#[test]
fn an_encode_item_is_steered_among_the_encoders_the_plan_admits() {
    let engine = PolicySpec::epd_presets("round_robin", "round_robin", "round_robin")
        .expect("the presets' form")
        .validate()
        .expect("the presets build");
    let mut pool = [over(0, 0), over(1, 4), over(2, 2)];
    pool[1].ec_compatible = true;
    pool[2].ec_compatible = true;
    let item = RequestFacts {
        phase: Phase::Encode,
        ..facts(1)
    };
    assert_eq!(order(&decide(&engine, &item, &pool)), vec![2, 1]);
}
