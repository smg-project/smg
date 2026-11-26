use super::*;

/// A pair's trace is its prefill's: the prefill profile's index and the
/// rows that profile's filters left of the prefill pool. The decode ran
/// another profile over another pool, and its trace is not the pair's.
/// Each target names the prefill with a decode, and no session verdict
/// stands on a pair.
#[test]
fn a_pairs_trace_is_its_prefills() {
    let engine = Engine::new(
        vec![least_requests_profile(), least_requests_profile()],
        vec![Binding {
            decode_profile: Some(1),
            ..pair_binding()
        }],
        1,
        Needs::default(),
    );
    let decision = decide_pair(&engine, &paired(3), &paired(2));
    let pick = decision.pick().expect("a pair");
    assert_eq!(pick.trace.profile, 0);
    assert_eq!(pick.trace.survivors.get(), Some(3));
    assert!(!pick.targets.is_empty());
    assert!(pick.targets.iter().all(|target| target.pair.is_some()));
    assert_eq!(pick.session_verdict, SessionVerdict::None);
}

/// A pair's decision with `scratch`, which keeps the decision's word of
/// the list it shed over.
fn decide_pair_with(
    engine: &Engine,
    prefills: &[Candidate],
    decodes: &[Candidate],
    scratch: &mut Scratch,
) -> Decision {
    decide_pair_at(engine, prefills, decodes, [None; 2], scratch)
}

/// [`decide_pair_with`] for a request held to places of each list: the
/// prefills', then the decodes', and `None` for every place.
fn decide_pair_at(
    engine: &Engine,
    prefills: &[Candidate],
    decodes: &[Candidate],
    places: [Option<&[u32]>; 2],
    scratch: &mut Scratch,
) -> Decision {
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let pools = PairPools {
        prefills,
        decodes,
        ..PairPools::default()
    };
    let [prefill_cx, decode_cx] = places.map(|places| Constraints {
        places,
        ..Constraints::default()
    });
    let pool = PoolFacts::default();
    engine.decide_pair(&facts, &pool, pools, &prefill_cx, &decode_cx, scratch)
}

/// A row of a pair's pool: of pair class `class`, over its thresholds
/// where `over`, and not healthy where `down`.
fn pair_row(position: u32, class: u32, over: bool, down: bool) -> Candidate {
    let mut row = candidate(position, 0);
    row.pair_class = class;
    row.health.overloaded = over;
    row.health.healthy = !down;
    row
}

/// A pair's decision says the list it sheds its request over as
/// saturated, under a binding that sheds a list wholly over its
/// thresholds. The prefills, where no prefill stands but for the veto.
/// The decodes, where the whole decode list is over its thresholds. No
/// list where a decode that pairs with no prefill is under them: the
/// shed is then made over the partners alone. Under a binding that
/// steers the same sheds say no list.
#[test]
fn a_pairs_decision_says_the_list_it_sheds_over() {
    let steering = || {
        Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        )
    };
    let shedding = || {
        let mut engine = steering();
        engine.shed_when_all_vetoed(0);
        engine
    };
    let saturated = Some((ShedReason::Saturated, "not-overloaded"));
    // The prefill is over its thresholds beside a decode that is down:
    // no pair is left to steer to, and the shed is the prefills'.
    let prefills = [pair_row(0, 7, true, false)];
    let decodes = [pair_row(0, 7, false, true)];
    for (engine, over) in [(shedding(), Some(ShedOver::Prefills)), (steering(), None)] {
        let mut scratch = Scratch::new();
        let decision = decide_pair_with(&engine, &prefills, &decodes, &mut scratch);
        assert_eq!(shed_of(&decision), saturated);
        assert_eq!(scratch.shed_over(), over, "the prefills");
    }
    // Every decode is over its thresholds beside a prefill that stands.
    let prefills = [pair_row(0, 7, false, false)];
    let decodes = [pair_row(0, 7, true, false), pair_row(1, 7, true, false)];
    let mut scratch = Scratch::new();
    let decision = decide_pair_with(&shedding(), &prefills, &decodes, &mut scratch);
    assert_eq!(shed_of(&decision), saturated);
    assert_eq!(scratch.shed_over(), Some(ShedOver::Decodes), "the decodes");
    // The prefill's partner is over its thresholds beside an idle decode
    // of a class no prefill pairs with: the decode list is not wholly
    // over them, under either binding.
    let decodes = [pair_row(0, 7, true, false), pair_row(1, 9, false, false)];
    for engine in [shedding(), steering()] {
        let mut scratch = Scratch::new();
        let decision = decide_pair_with(&engine, &prefills, &decodes, &mut scratch);
        assert_eq!(shed_of(&decision), saturated);
        assert_eq!(scratch.shed_over(), None, "the partners alone");
    }
}

/// A pair's decision says a list where the gateway's own selection
/// would read it as wholly over its thresholds, and no list where the
/// two counts part ([`vetoed::wholly_over`]). A worker under its thresholds
/// and under a liveness doubt keeps its list from being wholly over
/// them, though no pass of the decision keeps it: a decode that pairs
/// with no prefill for the decodes, a prefill for the prefills. Over
/// its thresholds too, it does not. A prefill that pairs with no
/// decode is no part of the prefills' count: the one over its
/// thresholds beside a partnered prefill that is down shows no
/// overload of the list, and beside a partnered one over its
/// thresholds it changes nothing.
///
/// That selection counts a worker that is healthy with a closed
/// breaker, whatever doubt stands on it
/// (`model_gateway/src/routers/common/overload.rs:58-74`), and holds
/// the prefills that pair with a decode alone
/// (`model_gateway/src/routers/common/pair_index.rs:158-161`, on main
/// `56773d28d`).
#[test]
fn a_pairs_decision_says_a_list_by_the_gateways_own_count() {
    let mut engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    engine.shed_when_all_vetoed(0);
    let doubted = |mut row: Candidate| {
        row.health.stalled = true;
        row
    };
    let said = |prefills: &[Candidate], decodes: &[Candidate]| {
        let mut scratch = Scratch::new();
        let decision = decide_pair_with(&engine, prefills, decodes, &mut scratch);
        assert_eq!(
            shed_of(&decision),
            Some((ShedReason::Saturated, "not-overloaded")),
            "{decision:?}"
        );
        scratch.shed_over()
    };
    let open = pair_row(0, 7, false, false);
    let over = pair_row(0, 7, true, false);
    let down = pair_row(0, 7, false, true);
    // The partner is over its thresholds beside a decode of a class no
    // prefill pairs with, which is under a doubt.
    let idle = doubted(pair_row(1, 9, false, false));
    assert_eq!(said(&[open], &[over, idle]), None, "a decode under a doubt");
    let busy = doubted(pair_row(1, 9, true, false));
    assert_eq!(said(&[open], &[over, busy]), Some(ShedOver::Decodes));
    // One prefill is over its thresholds beside one under a doubt, and
    // the decode is down.
    let idle = doubted(pair_row(1, 7, false, false));
    assert_eq!(
        said(&[over, idle], &[down]),
        None,
        "a prefill under a doubt"
    );
    let busy = doubted(pair_row(1, 7, true, false));
    assert_eq!(said(&[over, busy], &[down]), Some(ShedOver::Prefills));
    // The one prefill over its thresholds is of a class no decode is of.
    let alone = pair_row(1, 9, true, false);
    assert_eq!(
        said(&[down, alone], &[open]),
        None,
        "a prefill with no decode"
    );
    assert_eq!(said(&[over, alone], &[open]), Some(ShedOver::Prefills));
}

/// A pair's decision counts a list at the request's places alone. The
/// first prefill is over its thresholds and is the one place the
/// request may take: the decision says the prefills, though the second
/// prefill is under its thresholds. So it is for the decodes. Held to
/// no place, the same lists hold a pair that takes the request.
#[test]
fn a_pairs_decision_counts_a_list_at_the_requests_places() {
    let mut engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    engine.shed_when_all_vetoed(0);
    let said = |prefills: &[Candidate], decodes: &[Candidate], places: [Option<&[u32]>; 2]| {
        let mut scratch = Scratch::new();
        let decision = decide_pair_at(&engine, prefills, decodes, places, &mut scratch);
        (shed_of(&decision), scratch.shed_over())
    };
    let saturated = Some((ShedReason::Saturated, "not-overloaded"));
    let (open, over) = (pair_row(0, 7, false, false), pair_row(0, 7, true, false));
    let second = pair_row(1, 7, false, false);
    let first: &[u32] = &[0];
    assert_eq!(
        said(&[over, second], &[open], [Some(first), None]),
        (saturated, Some(ShedOver::Prefills))
    );
    assert_eq!(
        said(&[open], &[over, second], [None, Some(first)]),
        (saturated, Some(ShedOver::Decodes))
    );
    assert_eq!(said(&[over, second], &[open], [None; 2]), (None, None));
    assert_eq!(said(&[open], &[over, second], [None; 2]), (None, None));
}

/// The word of the list a shed was made over is its decision's alone:
/// the next pair's decision made with the same scratch takes it back
/// where it begins, whether it picks, sheds for another reason or is
/// declined ahead of every profile.
#[test]
fn a_pairs_decision_takes_back_the_word_of_the_one_before() {
    let mut engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    engine.shed_when_all_vetoed(0);
    let over = [pair_row(0, 7, true, false)];
    let open = [pair_row(0, 7, false, false)];
    let down = [pair_row(0, 7, false, true)];
    let mut scratch = Scratch::new();
    let shed_over_the_prefills = |scratch: &mut Scratch| {
        let decision = decide_pair_with(&engine, &over, &open, scratch);
        assert_eq!(
            scratch.shed_over(),
            Some(ShedOver::Prefills),
            "{decision:?}"
        );
    };
    // A decision that picks.
    shed_over_the_prefills(&mut scratch);
    let decision = decide_pair_with(&engine, &open, &open, &mut scratch);
    assert!(decision.pick().is_some(), "{decision:?}");
    assert_eq!(scratch.shed_over(), None, "behind a pick");
    // A decision that sheds with no endpoint.
    shed_over_the_prefills(&mut scratch);
    let decision = decide_pair_with(&engine, &down, &open, &mut scratch);
    assert_eq!(
        shed_of(&decision),
        Some((ShedReason::NoEndpoints, "healthy"))
    );
    assert_eq!(scratch.shed_over(), None, "behind another shed");
    // A decision declined ahead of every profile: no binding matches.
    shed_over_the_prefills(&mut scratch);
    let unbound = Engine::new(vec![strict_profile()], vec![], 1, Needs::default());
    let decision = decide_pair_with(&unbound, &over, &open, &mut scratch);
    assert_eq!(
        shed_of(&decision),
        Some((ShedReason::PolicyDeclined, "binding"))
    );
    assert_eq!(scratch.shed_over(), None, "behind a decline");
}

/// The side of a pair a shed names ([`Shed::side`]).
fn side_of(decision: &Decision) -> Option<PairSide> {
    match decision {
        Decision::Shed(shed) => shed.side,
        other => panic!("not shed: {other:?}"),
    }
}

/// A pair's refusal names the side that was left with nobody. The
/// prefill side where no prefill of the request's stands: none is
/// listed, each is down, or each is over its thresholds with nowhere to
/// be steered. The decode side where a prefill stands and its partners
/// are missing or refused: no decode is of its class, none is listed,
/// each is down, or each is over its thresholds under a binding that
/// sheds. The decode side too where the prefill was picked and the
/// decode's picker declined. A pair declined ahead of every profile
/// names no side: neither list was read.
///
/// A decode decided again names the decode side for the partners it
/// lacks, and the prefill side for a held prefill that left its list.
///
/// The gateway's own pair selection names a role in each refusal and
/// reads the prefills first: a pool with no worker
/// (`model_gateway/src/routers/common/placement.rs:2860-2867`), the
/// prefills with nobody open (`:2900-2918`), then the decodes, which
/// left the open prefills nothing to pair with (`:2943-2970`), and the
/// decode policy's own decline (`:3090`), on main `d43c7b2a9`. It
/// decides no decode again.
#[test]
fn a_pairs_refusal_names_the_side_left_with_nobody() {
    let steering = strict_pair_engine();
    let mut shedding = strict_pair_engine();
    shedding.shed_when_all_vetoed(0);
    let open = pair_row(0, 7, false, false);
    let over = pair_row(0, 7, true, false);
    let down = pair_row(0, 7, false, true);
    let apart = pair_row(0, 9, false, false);
    let names = |decision: &Decision| {
        let stage = shed_of(decision).map(|(_, stage)| stage);
        (stage, side_of(decision))
    };
    let said = |engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]| {
        names(&decide_pair(engine, prefills, decodes))
    };
    let (prefill, decode) = (Some(PairSide::Prefill), Some(PairSide::Decode));
    // No prefill of the request's stands.
    assert_eq!(said(&steering, &[], &[open]), (Some("none"), prefill));
    assert_eq!(
        said(&steering, &[down], &[open]),
        (Some("healthy"), prefill)
    );
    let vetoed = Some("not-overloaded");
    assert_eq!(said(&steering, &[over], &[down]), (vetoed, prefill));
    assert_eq!(said(&shedding, &[over], &[open]), (vetoed, prefill));
    // Both lists are down: the prefills are read first.
    assert_eq!(
        said(&steering, &[down], &[down]),
        (Some("healthy"), prefill)
    );
    // A prefill stands, and its partners are missing or refused. The
    // last of these prefills is of no pair class: it pairs with nobody.
    let unpartnered = Some("partnered");
    assert_eq!(said(&steering, &[open], &[apart]), (unpartnered, decode));
    assert_eq!(said(&steering, &[open], &[]), (unpartnered, decode));
    let classless = pair_row(0, 0, false, false);
    assert_eq!(
        said(&steering, &[classless], &[open]),
        (unpartnered, decode)
    );
    assert_eq!(said(&steering, &[open], &[down]), (Some("healthy"), decode));
    assert_eq!(said(&shedding, &[open], &[over]), (vetoed, decode));
    // The prefill is picked, and the decode's own picker declines.
    let declining = Profile {
        name: "decode".into(),
        filters: vec![Box::new(Healthy)],
        scorers: vec![],
        picker: Box::new(RefusesClass(7)),
    };
    let binding = Binding {
        decode_profile: Some(1),
        ..pair_binding()
    };
    let profiles = vec![strict_profile(), declining];
    let declined = Engine::new(profiles, vec![binding], 1, Needs::default());
    assert_eq!(
        said(&declined, &[open], &[open]),
        (Some("refuses-class"), decode)
    );
    // No binding: the pair is declined ahead of either list.
    let unbound = Engine::new(vec![strict_profile()], vec![], 1, Needs::default());
    assert_eq!(said(&unbound, &[open], &[open]), (Some("binding"), None));

    // The decode decided again behind a held prefill.
    let again = |prefills: &[Candidate], decodes: &[Candidate], held: WorkerId| {
        let pools = PairPools {
            prefills,
            decodes,
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        names(&steering.decide_decode(&request(7), &pool, pools, held, None, &cx, &mut scratch))
    };
    let gone = WorkerId(open.id.0 + 1);
    assert_eq!(
        again(&[open], &[open], gone),
        (Some("prefill-gone"), prefill)
    );
    assert_eq!(again(&[open], &[down], open.id), (Some("healthy"), decode));
}

/// Where a pair's decision says the list it shed over, its refusal
/// names that list's side: the prefills are the prefill side, and the
/// decodes the decode side. The refusal names its side where the
/// decision says no list too: under a binding that steers, where the
/// decision's count and the gateway's own part, and for a shed with no
/// endpoint. So the side is not read from that word.
#[test]
fn a_pairs_refusal_names_the_side_of_the_list_its_decision_says() {
    let steering = strict_pair_engine();
    let mut shedding = strict_pair_engine();
    shedding.shed_when_all_vetoed(0);
    let said = |engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]| {
        let mut scratch = Scratch::new();
        let decision = decide_pair_with(engine, prefills, decodes, &mut scratch);
        (side_of(&decision), scratch.shed_over())
    };
    let open = pair_row(0, 7, false, false);
    let over = pair_row(0, 7, true, false);
    let down = pair_row(0, 7, false, true);
    let (prefill, decode) = (Some(PairSide::Prefill), Some(PairSide::Decode));
    // The two sites that say a list.
    let prefills = Some(ShedOver::Prefills);
    assert_eq!(said(&shedding, &[over], &[down]), (prefill, prefills));
    let decodes = Some(ShedOver::Decodes);
    assert_eq!(said(&shedding, &[open], &[over]), (decode, decodes));
    // Under a binding that steers the prefills' shed says no list.
    assert_eq!(said(&steering, &[over], &[down]), (prefill, None));
    // The partner is over its thresholds beside an idle decode that
    // pairs with no prefill: the two counts part, and no list is said.
    let idle = pair_row(1, 9, false, false);
    assert_eq!(said(&shedding, &[open], &[over, idle]), (decode, None));
    // A shed with no endpoint says no list.
    assert_eq!(said(&shedding, &[down], &[open]), (prefill, None));
    assert_eq!(said(&shedding, &[open], &[down]), (decode, None));
}

#[test]
fn a_prefill_whose_partners_fail_the_decode_filters_is_dropped_as_unpartnered() {
    // The decode side never runs for a prefill without a kept partner:
    // the shed names the decode stage that dropped the partners, or
    // "partnered" when decodes survive and none has a prefill's class.
    let engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 0)];
    prefills[0].pair_class = 7;
    let mut decodes = [candidate(0, 0)];
    decodes[0].pair_class = 7;
    decodes[0].health.healthy = false;
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(
                shed.emptied_by, "healthy",
                "the decode stage that dropped the partner"
            );
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("an unroutable partner answered {other:?}"),
    }
    decodes[0].health.healthy = true;
    decodes[0].pair_class = 8;
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "partnered", "no decode of the class");
        }
        other => panic!("an empty class answered {other:?}"),
    }
}

#[test]
fn a_pair_with_no_decode_at_all_is_dropped_as_unpartnered() {
    // An empty decode pool is a class with no decode for every prefill:
    // the prefill's own "partnered" shed stands, where the decode side's
    // filter alone would name "none".
    let engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 0)];
    prefills[0].pair_class = 7;
    match decide_pair(&engine, &prefills, &[]) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "partnered", "no decode of any class");
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("an empty decode pool answered {other:?}"),
    }
}

#[test]
fn the_pair_takes_its_decode_profile_from_the_binding_the_prefill_resolved() {
    // A decode-phase binding to the strict profile sits before the pair
    // binding, whose decode side is lenient; the pair uses its own.
    let lenient = Profile {
        name: "lenient".into(),
        filters: vec![Box::new(Healthy)],
        scorers: vec![],
        picker: Box::new(Best),
    };
    let engine = Engine::new(
        vec![lenient, strict_profile()],
        vec![
            Binding {
                model: "*".into(),
                phases: vec![Phase::Decode],
                plan: Plan::Single,
                profile: 1,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions: None,
            },
            pair_binding(),
        ],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 0)];
    prefills[0].pair_class = 7;
    let mut decodes = [candidate(0, 0)];
    decodes[0].pair_class = 7;
    decodes[0].health.overloaded = true;
    let decision = decide_pair(&engine, &prefills, &decodes);
    assert!(
        decision.pick().is_some(),
        "the pair binding's lenient decode side keeps the overloaded decode: {decision:?}"
    );
}

#[test]
fn a_pair_that_exists_is_found_past_the_prefill_pickers_targets() {
    // Five tied prefills and one decode, partnered with the fifth alone:
    // the prefill pool narrows to that one before the picker's four.
    let engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 0); 5];
    for (i, p) in prefills.iter_mut().enumerate() {
        p.position = i as u32;
        p.id = WorkerId(100 + i as u64);
        p.pair_class = i as u32;
    }
    let mut decodes = [candidate(0, 0)];
    decodes[0].pair_class = 4;
    for seed in 0..20u64 {
        let facts = RequestFacts {
            model: "m",
            seed,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("the partnered prefill is found");
        assert_eq!(
            pick.targets.first().map(|t| t.position),
            Some(4),
            "seed {seed}"
        );
    }
    decodes[0].health.healthy = false;
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(
                shed.emptied_by, "healthy",
                "the decode stage that left no partner"
            );
        }
        other => panic!("a pool without a partnered prefill answered {other:?}"),
    }
}

#[test]
fn a_single_binding_declines_a_pair_decision() {
    let prefills = [candidate(0, 0)];
    let decodes = [candidate(0, 0)];
    match decide_pair(&strict_engine(), &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(shed.reason, ShedReason::PolicyDeclined);
            assert_eq!(shed.emptied_by, "binding");
            assert_eq!(shed.trace.survivors.get(), Some(0), "no pool was read");
            assert_eq!(shed.trace.made_under(), None);
        }
        other => panic!("a single binding answered a pair with {other:?}"),
    }
}

/// A row whose position is not its index is a host's mistake. A debug
/// build stops at the assertion where a decision takes its inputs; the
/// assertion reads every row, and the misplaced one here is the second.
/// A release build keeps the pair plan's guard: the lookup of the chosen
/// prefill finds no row at that position and the target loses its turn,
/// never the process, and never to another row's class. The first pool
/// holds the last clause: its first prefill is of the decode's class and
/// busier, so the plan chooses the misplaced one and keeps the first as
/// its alternate; the misplaced one loses its turn and the alternate
/// leads, where a lookup that fell back to any row would pair the
/// misplaced one. The second pool has no alternate, and the decision is
/// shed. No hosted job builds these cases for release: the release side
/// runs by hand with every change of the pair entry, until the last
/// reader of a row's position goes and the assertion with it.
#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "a row's position is its index")
)]
fn a_misplaced_prefill_stops_a_debug_build_and_costs_a_release_build_its_target() {
    let engine = Engine::new(
        vec![least_requests_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let mut decodes = [candidate(0, 0)];
    decodes[0].pair_class = 7;
    // Both prefills pair with the decode and the first is busier: the
    // plan chooses the second, the misplaced one, then the first.
    let mut prefills = [candidate(0, 9), candidate(1, 0)];
    prefills[0].pair_class = 7;
    prefills[1].position = 5;
    prefills[1].pair_class = 7;
    let decision = decide_pair(&engine, &prefills, &decodes);
    assert_eq!(
        lead(&decision),
        Some(0),
        "the alternate leads where the misplaced prefill lost its turn: {decision:?}"
    );
    // The first prefill outside any pairing: the misplaced one is the
    // only target, and its lost turn sheds the decision.
    prefills[0].pair_class = 0;
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => assert_eq!(shed.emptied_by, "partners"),
        other => panic!("a misplaced prefill answered {other:?}"),
    }
}

#[test]
fn class_zero_is_outside_any_pairing_on_either_side() {
    // A host that leaves the classes unset: a class-0 prefill over a
    // healthy class-0 decode sheds at the partnered stage instead of
    // pairing all with all.
    let engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let prefills = [candidate(0, 0)];
    let decodes = [candidate(0, 0)];
    match decide_pair(&engine, &prefills, &decodes) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "partnered");
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("two class-0 rows answered {other:?}"),
    }
    // A prefill of a class over decodes of class 0 alone: no partner.
    let mut classed = [candidate(0, 0)];
    classed[0].pair_class = 7;
    match decide_pair(&engine, &classed, &decodes) {
        Decision::Shed(shed) => assert_eq!(shed.emptied_by, "partnered"),
        other => panic!("a class-0 decode partnered {other:?}"),
    }
}

#[test]
fn a_class_zero_prefill_is_passed_over_for_a_classed_one() {
    // A just-admitted prefill with no class yet beside a classed one
    // whose decode is up: the pair is the classed one's, for every seed,
    // and the class-0 decode is no alternate.
    let engine = Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 0), candidate(1, 0)];
    prefills[1].pair_class = 7;
    let mut decodes = [candidate(0, 0), candidate(1, 0)];
    decodes[1].pair_class = 7;
    for seed in 0..20u64 {
        let facts = RequestFacts {
            model: "m",
            seed,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("the classed prefill pairs");
        let first = pick.targets.first().expect("a target");
        assert_eq!(first.position, 1, "seed {seed}");
        assert_eq!(
            first.pair.map(|p| p.decode_position),
            Some(1),
            "seed {seed}"
        );
        assert_eq!(pick.targets.len(), 1, "the class-0 decode is no alternate");
    }
}

/// A pair's decode profile is a decode row of the pair binding's glob,
/// whatever phases the binding covers: an engine built by hand may bind
/// a pair for the prefill phase alone, and a decode decision of that
/// model carries the pair's decode profile. A pair bound off the prefill
/// phase starts no pair: it writes no decode row.
#[test]
fn a_pairs_decode_profile_is_a_needs_row_of_the_pairs_glob() {
    #[derive(Debug)]
    struct Fed;

    impl Filter for Fed {
        fn name(&self) -> &'static str {
            "fed"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn needs(&self) -> Needs {
            Needs {
                loads: true,
                ..Needs::default()
            }
        }
    }

    let quiet = least_requests_profile();
    let fed = Profile {
        filters: vec![Box::new(Fed)],
        ..least_requests_profile()
    };
    let decode = fed.needs();
    assert_eq!(quiet.needs(), Needs::default());
    assert!(decode.loads);
    let engine = Engine::new(
        vec![quiet, fed],
        vec![Binding {
            model: "pd/*".into(),
            phases: vec![Phase::Prefill],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(1),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    assert_eq!(engine.needs_for("pd/x", Phase::Decode), decode);
    assert_eq!(engine.needs_rows(), [("pd/*", Phase::Decode, decode)]);
    // An `epd` pair's encode profile is an encode row of the glob the
    // same way: an encode decision of the model carries it, and so
    // does the encode phase's own answer.
    let fed = Profile {
        filters: vec![Box::new(Fed)],
        ..least_requests_profile()
    };
    let engine = Engine::new(
        vec![least_requests_profile(), fed],
        vec![Binding {
            model: "pd/*".into(),
            phases: vec![Phase::Prefill],
            plan: Plan::Epd { encode: 1 },
            profile: 0,
            decode_profile: Some(0),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let encode = decode;
    assert_eq!(engine.needs_for("pd/x", Phase::Encode), encode);
    assert_eq!(engine.needs_for_phase(Phase::Encode), encode);
    assert_eq!(engine.needs_for("pd/x", Phase::Decode), Needs::default());
    assert_eq!(engine.needs_rows(), [("pd/*", Phase::Encode, encode)]);
    // No prefill lookup resolves to a pair bound for another phase.
    let fed = Profile {
        filters: vec![Box::new(Fed)],
        ..least_requests_profile()
    };
    let engine = Engine::new(
        vec![least_requests_profile(), fed],
        vec![Binding {
            model: "pd/*".into(),
            phases: vec![Phase::Encode],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(1),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    assert_eq!(engine.needs_for("pd/x", Phase::Decode), Needs::default());
    assert!(engine.needs_rows().is_empty(), "{:?}", engine.needs_rows());
}

#[test]
fn the_pair_plan_pairs_the_prefill_with_a_decode_of_its_class() {
    let engine = Engine::new(
        vec![Profile {
            name: "least".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![WeightedScorer {
                scorer: Box::new(LeastRequests),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        }],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(0),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 2), candidate(1, 0)];
    prefills[0].pair_class = 1;
    prefills[1].pair_class = 2;
    let mut decodes = [candidate(0, 0), candidate(1, 7), candidate(2, 1)];
    decodes[0].pair_class = 1;
    decodes[1].pair_class = 2;
    decodes[2].pair_class = 2;
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide_pair(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut scratch,
    );
    let pick = decision.pick().expect("a pair");
    let first = pick.targets.first().expect("a target");
    assert_eq!(first.position, 1, "the idle prefill");
    assert_eq!(
        first.pair.map(|p| p.decode_position),
        Some(2),
        "its least-loaded partner"
    );
    assert_eq!(
        pick.targets.len(),
        2,
        "the alternate decode of the same prefill"
    );
    assert_eq!(
        pick.targets.as_slice()[1].pair.map(|p| p.decode_position),
        Some(1)
    );
}

#[test]
fn the_decode_picker_does_not_replay_the_prefill_pickers_draws() {
    // Random on both sides, four prefills of four decodes each. With one
    // seed on both sides the decode index inside the class followed the
    // prefill index, and four of the sixteen decodes took every pair.
    let engine = PolicySpec::pair_presets("random", "random")
        .expect("the pair spec")
        .validate()
        .expect("the engine");
    let mut prefills = [candidate(0, 0); 4];
    for (i, p) in prefills.iter_mut().enumerate() {
        p.position = i as u32;
        p.id = WorkerId(100 + i as u64);
        p.pair_class = i as u32 + 1;
        p.has_capacity = true;
    }
    let mut decodes = [candidate(0, 0); 16];
    for (i, d) in decodes.iter_mut().enumerate() {
        d.position = i as u32;
        d.id = WorkerId(200 + i as u64);
        d.pair_class = i as u32 / 4 + 1;
        d.has_room = true;
    }
    let mut picked = [0u32; 16];
    for seed in 0..4096u64 {
        let facts = RequestFacts {
            model: "m",
            seed,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a pair");
        let first = pick.targets.first().expect("a target");
        let decode = first.pair.expect("a decode").decode_position as usize;
        assert_eq!(
            decodes[decode].pair_class, prefills[first.position as usize].pair_class,
            "seed {seed}: the decode is of the prefill's class"
        );
        picked[decode] += 1;
    }
    let idle: Vec<usize> = picked
        .iter()
        .enumerate()
        .filter(|(_, &n)| n == 0)
        .map(|(i, _)| i)
        .collect();
    assert!(
        idle.is_empty(),
        "decodes never picked over 4096 seeds: {idle:?}"
    );
}

#[test]
fn a_prefill_without_partners_is_passed_over_for_the_next() {
    let engine = Engine::new(
        vec![least_requests_profile()],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(0),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let mut prefills = [candidate(0, 0), candidate(1, 5)];
    prefills[0].pair_class = 7;
    prefills[1].pair_class = 8;
    let mut decodes = [candidate(0, 0)];
    decodes[0].pair_class = 8;
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide_pair(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut scratch,
    );
    let pick = decision.pick().expect("the busier prefill still pairs");
    assert_eq!(pick.targets.first().map(|t| t.position), Some(1));
}

/// Under the pair plan the prefill side's targets are walked until one
/// pairs. A narrowing picker keeps the survivors it passed over as
/// alternates, so when the decode side declines the partners of the
/// prefill compared, a prefill passed over still pairs: a pair that was a
/// pick before the narrowing is one after it.
#[test]
fn a_prefill_the_picker_passed_over_still_pairs_when_the_one_compared_cannot() {
    // The idle prefill's decode is refused; the busy prefill's is not.
    let mut prefills = [candidate(0, 0), candidate(1, 5)];
    prefills[0].pair_class = 7;
    prefills[1].pair_class = 8;
    let mut decodes = [candidate(0, 0), candidate(1, 0)];
    decodes[0].pair_class = 7;
    decodes[1].pair_class = 8;
    let pickers: [fn() -> Box<dyn Picker>; 3] = [
        || Box::new(CountFirst),
        || Box::new(PowerOfTwo),
        || Box::new(PowerOfTwoCountFirst),
    ];
    for picker in pickers {
        let picker = picker();
        let name = picker.name();
        let engine = Engine::new(
            vec![
                Profile {
                    name: "prefill".into(),
                    filters: vec![Box::new(Healthy)],
                    scorers: vec![WeightedScorer {
                        scorer: Box::new(LeastRequests),
                        weight: 1.0,
                    }],
                    picker,
                },
                Profile {
                    name: "decode".into(),
                    filters: vec![Box::new(Healthy)],
                    scorers: vec![],
                    picker: Box::new(RefusesClass(7)),
                },
            ],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(1),
                shadow: None,
                decode_shadow: None,
                sessions: None,
            }],
            1,
            Needs::default(),
        );
        for seed in 0..32u64 {
            let facts = RequestFacts {
                model: "m",
                seed,
                ..RequestFacts::default()
            };
            let decision = engine.decide_pair(
                &facts,
                &PoolFacts::default(),
                PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    ..PairPools::default()
                },
                &Constraints::default(),
                &Constraints::default(),
                &mut Scratch::new(),
            );
            let Some(pick) = decision.pick() else {
                panic!("{name}, seed {seed}: no pair in {decision:?}")
            };
            let first = pick.targets.first().expect("a target");
            assert_eq!(
                (first.position, first.pair.map(|pair| pair.decode_position)),
                (1, Some(1)),
                "{name}, seed {seed}"
            );
        }
    }
}

/// Under the pair plan the decode side opens a stream of its own from
/// the request's seed. The order count-first gives equal counts among
/// the prefills it passed over shares no value with that stream: when
/// the lead prefill's decode run sheds, each of two tied alternates
/// pairs with each of its two decodes about as often. (Ordered by the
/// decode side's own stream, two of the four pairs are never reached.)
#[test]
fn a_passed_over_prefill_pairs_with_every_decode_of_its_class() {
    /// The `random` picker, unless the first survivor is of the refused
    /// pair class: then the picker declines.
    #[derive(Debug)]
    struct RandomUnlessClass(u32);
    impl Lifecycle for RandomUnlessClass {}
    impl Picker for RandomUnlessClass {
        fn name(&self) -> &'static str {
            "random-unless-class"
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
            let first = kept.first().map(|&row| candidates[row as usize].pair_class);
            if first != Some(self.0) {
                Random.pick(facts, candidates, kept, scores, band, out);
            }
        }
    }
    // The idle prefill leads and its decode side declines; the two busy
    // prefills tie and hold a class of two decodes each.
    let mut prefills = [candidate(0, 0), candidate(1, 5), candidate(2, 5)];
    for (prefill, class) in prefills.iter_mut().zip([7, 8, 9]) {
        prefill.pair_class = class;
    }
    let mut decodes = [
        candidate(0, 0),
        candidate(1, 0),
        candidate(2, 0),
        candidate(3, 0),
        candidate(4, 0),
    ];
    for (decode, class) in decodes.iter_mut().zip([7, 8, 8, 9, 9]) {
        decode.pair_class = class;
    }
    let engine = Engine::new(
        vec![
            Profile {
                name: "prefill".into(),
                filters: vec![Box::new(Healthy)],
                scorers: vec![WeightedScorer {
                    scorer: Box::new(LeastRequests),
                    weight: 1.0,
                }],
                picker: Box::new(CountFirst),
            },
            Profile {
                name: "decode".into(),
                filters: vec![Box::new(Healthy)],
                scorers: vec![],
                picker: Box::new(RandomUnlessClass(7)),
            },
        ],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(1),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    // How often each busy prefill pairs with each decode of its class.
    let mut pairs = [[0u32; 2]; 2];
    for seed in 0..4_000u64 {
        let facts = RequestFacts {
            model: "m",
            seed,
            ..RequestFacts::default()
        };
        let decision = engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let Some(pick) = decision.pick() else {
            panic!("seed {seed}: no pair in {decision:?}")
        };
        let first = pick.targets.first().expect("a target");
        let decode = first.pair.expect("a pair").decode_position;
        assert!(
            (1..=2).contains(&first.position) && (1..=4).contains(&decode),
            "seed {seed}: {first:?}"
        );
        pairs[(first.position - 1) as usize][((decode - 1) % 2) as usize] += 1;
    }
    assert!(
        pairs.iter().flatten().all(|n| (850..=1150).contains(n)),
        "{pairs:?}"
    );
}
