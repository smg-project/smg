//! The rule for a pair's two lists: the pair's default, each side's own
//! facts, first partners, and a decode asked again behind its prefill.

use super::*;

/// A pair's prefill pool nobody is usable in is shed by the stage that
/// refuses its workers. The prefill side's own run is put to no rule,
/// and the pair's default asks the profile's filters once more with
/// each veto lifted: the filter listed behind the veto refuses both
/// prefills in each of its two passes. Under a binding that steers no
/// such shed is saturated: the veto ended the run, the prefills are
/// wholly over their thresholds, and the shed names the stage that
/// refuses the first of them with the veto set aside
/// ([`Half::refused_room`]), which is asked of that one row once more.
#[test]
fn a_pairs_prefill_pool_with_no_usable_worker_sheds_at_the_stage_behind_the_veto() {
    let none: (&[u32], &[u32]) = (&[], &[]);
    let side = pairs_side_shed(Box::new(NotOverloaded), [over(0, 1), over(1, 2)], none);
    assert_eq!(side, ((ShedReason::NoEndpoints, "refuses"), 5));
}

/// Under a binding that steers, a pair's decode side over its thresholds
/// on every worker goes to the decode that holds the fewest requests
/// among the chosen prefill's partners. No decode is open, so the rule
/// first says which decodes it keeps and the prefill is chosen for
/// them: a prefill whose class holds no decode is passed over. A doubted
/// decode among them is not kept, however few requests it holds. The
/// gateway's own pair selection answers so
/// (`model_gateway/src/routers/common/placement.rs:1315-1321`,
/// `:1338-1341`, `:1571-1578`).
#[test]
fn a_pairs_decode_side_over_on_every_worker_goes_to_the_fewest_in_flight() {
    let engine = pair_engine(false);
    let prefills = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    let mut decodes = [over(0, 3), over(1, 1), over(2, 2)].map(|worker| classed(worker, 1));
    let decision = decide_pair(&engine, &prefills, &decodes);
    assert_eq!(pair_of(&decision), (1, 1));
    assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
    decodes[1].health.stalled = true;
    let decision = decide_pair(&engine, &prefills, &decodes);
    assert_eq!(pair_of(&decision), (1, 2));
    assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
}

/// Under a binding that sheds, a pair's decode side over its thresholds
/// on every worker is shed as saturated, as the gateway's own selection
/// lifts nothing where it sheds (`placement.rs:1314-1321` there).
#[test]
fn a_binding_that_sheds_sheds_a_pairs_decode_side_over_on_every_worker() {
    let (prefills, decodes) = ([classed(free(0, 1), 1)], [classed(over(0, 1), 1)]);
    let decision = decide_pair(&pair_engine(true), &prefills, &decodes);
    assert_eq!(
        shed_of(&decision),
        (ShedReason::Saturated, "not-overloaded")
    );
}

/// Under a binding that sheds, a doubted decode under its thresholds
/// beside a busy one takes the pair. The rule keeps that decode past its
/// doubt, and a decode kept so opens no prefill: the prefill side sheds,
/// and the pair's default pairs the free prefill with it, the liveness
/// veto lifted and no worker over its thresholds taken. The gateway's own
/// pair selection places that pair by its fallback
/// (`model_gateway/src/routers/common/placement.rs:1443-1458`,
/// `:1138-1147`).
#[test]
fn a_binding_that_sheds_pairs_past_a_decodes_doubt() {
    let prefills = [classed(free(0, 0), 1)];
    let decodes = [classed(over(0, 1), 1), classed(doubted(1, 0), 1)];
    let decision = decide_pair(&pair_engine(true), &prefills, &decodes);
    assert_eq!(pair_of(&decision), (0, 1));
    assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
}

/// What a pair's first decision asks of its rule: the overload veto of
/// a decode pool, and no more. Decodes of one class all over stay
/// unasked while a decode of another class is open. A decode pool
/// doubted on every worker opens no prefill, and a prefill pool over on
/// every worker is put to no rule: each goes to the pair's default,
/// which says the veto it lifted.
#[test]
fn a_pairs_first_decision_answers_for_a_decode_pool_over_alone() {
    let engine = pair_engine(false);
    let prefills = [classed(free(0, 0), 1), classed(free(1, 5), 2)];
    let decodes = [
        classed(over(0, 0), 1),
        classed(over(1, 0), 1),
        classed(free(2, 9), 2),
    ];
    let decision = decide_pair(&engine, &prefills, &decodes);
    assert_eq!(pair_of(&decision), (1, 2));
    assert_eq!(decision.trace().lifted, LiftedVeto::None);
    let (idle, busy) = ([classed(free(0, 0), 1)], [classed(over(0, 0), 1)]);
    let suspect = [classed(doubted(0, 0), 1)];
    let (doubt, load) = (LiftedVeto::Liveness, LiftedVeto::Overload);
    for (prefills, decodes, lifted) in [(&idle, &suspect, doubt), (&busy, &idle, load)] {
        let paired = decide_pair(&engine, prefills, decodes);
        assert_eq!((pair_of(&paired), paired.trace().lifted), ((0, 0), lifted));
    }
}

/// A prefill pool refused on every worker goes to the pair of the least
/// load: the prefill's requests first, then its partner's. Pairs the
/// overload veto alone refuses come first: a doubted prefill that holds
/// fewer is passed by. A binding that sheds takes no worker over its
/// thresholds: the doubted prefill takes the request, and a pool over on
/// every worker is shed as saturated. The gateway's own default pairs so
/// (`model_gateway/src/routers/common/placement.rs:1093-1147`, asked at
/// `:1395-1413`).
#[test]
fn a_refused_prefill_pool_goes_to_the_pair_of_the_least_load() {
    let prefills = [
        classed(over(0, 2), 1),
        classed(over(1, 1), 2),
        classed(over(2, 1), 3),
        classed(doubted(3, 0), 3),
    ];
    let decodes = [free(0, 0), free(1, 5), free(2, 4)];
    let decodes = [0, 1, 2].map(|at| classed(decodes[at], at as u32 + 1));
    let steered = decide_pair(&pair_engine(false), &prefills, &decodes);
    assert_eq!(pair_of(&steered), (2, 2));
    // The pair's trace is the prefill side's: its profile, the veto
    // lifted, and the three prefills the lifted pass kept.
    let trace = steered.trace();
    assert_eq!((trace.profile, trace.lifted), (0, LiftedVeto::Overload));
    assert_eq!(trace.survivors, Survivors::counted(3));
    let shedding = pair_engine(true);
    let doubted = decide_pair(&shedding, &prefills, &decodes);
    assert_eq!(pair_of(&doubted), (3, 2));
    assert_eq!(doubted.trace().lifted, LiftedVeto::Liveness);
    let refused = decide_pair(&shedding, &prefills[..3], &decodes);
    assert_eq!(shed_of(&refused), (ShedReason::Saturated, "not-overloaded"));
}

/// A free prefill no decode pairs with is no prefill of the pool the
/// default judges: beside it, the prefills that have a partner, all over
/// their thresholds, pair by load. The gateway's index lists the
/// prefills with a compatible decode, and its default is judged over
/// them (`model_gateway/src/routers/common/pair_index.rs:99-103`,
/// `placement.rs:1395-1404` beside it).
#[test]
fn a_free_prefill_no_decode_pairs_with_is_not_of_the_refused_pool() {
    let prefills = [classed(over(0, 1), 1), classed(free(1, 0), 2)];
    let decodes = [classed(free(0, 0), 1)];
    let paired = decide_pair(&pair_engine(false), &prefills, &decodes);
    assert_eq!(pair_of(&paired), (0, 0));
    assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
}

/// The pair's default keeps each side to the places its host states. A
/// worker outside them is no worker of the pool the default judges,
/// however free, and no pass takes it: a free prefill outside the
/// places, and a free decode outside them. The gateway's own default
/// judges the workers the request may use and pairs among them
/// (`model_gateway/src/routers/common/placement.rs:774-786`,
/// `:1097-1104`).
///
/// The prefills it judges are those with a decode to pair with anywhere
/// in the pool. A free prefill whose one partner is outside the decodes'
/// places stands among them, so beside a free decode at the places the
/// request is refused, as on main: the gateway lists a prefill's
/// partners for the pool and no request (`pair_index.rs:99-103`,
/// `:143-146` there), finds no pool vetoed on every worker and refuses
/// too (`placement.rs:1438-1447`, `:1094-1096`).
#[test]
fn a_pairs_default_keeps_to_each_sides_places() {
    let engine = pair_engine(false);
    let held = |prefills: &[Candidate], decodes: &[Candidate], places: [Option<&[u32]>; 2]| {
        let pools = PairPools {
            prefills,
            decodes,
            ..PairPools::default()
        };
        let [cx, decode_cx] = places.map(|places| Constraints {
            places,
            ..Constraints::default()
        });
        let (req, pool, scratch) = (facts(1), PoolFacts::default(), &mut Scratch::new());
        let paired = engine.decide_pair(&req, &pool, pools, &cx, &decode_cx, scratch);
        let lifted = paired.trace().lifted;
        let pair = paired.pick().map(|_| (pair_of(&paired), lifted));
        pair.ok_or_else(|| shed_of(&paired))
    };
    let (first, every): (Option<&[u32]>, Option<&[u32]>) = (Some(&[0]), None);
    let prefills = [classed(over(0, 1), 1), classed(free(1, 0), 1)];
    let pair = held(&prefills, &[classed(free(0, 0), 1)], [first, every]);
    assert_eq!(pair, Ok(((0, 0), LiftedVeto::Overload)));
    let decodes = [classed(doubted(0, 1), 1), classed(free(1, 0), 1)];
    let pair = held(&[classed(free(0, 0), 1)], &decodes, [every, first]);
    assert_eq!(pair, Ok(((0, 0), LiftedVeto::Liveness)));
    let prefills = [classed(over(0, 1), 1), classed(free(1, 0), 2)];
    let decodes = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
    let refused = held(&prefills, &decodes, [every, first]);
    assert_eq!(refused, Err((ShedReason::NoEndpoints, "partnered")));
}

/// A pool the liveness veto refuses on every worker pairs by load under
/// either word of the binding: the prefill pool, and the decode pool
/// beside a standing prefill. A worker a hold stands on is never taken,
/// however few requests it holds. The gateway's own default lifts the
/// liveness veto so (`model_gateway/src/routers/common/placement.rs:1138-1147`,
/// asked at `:1395-1413` and `:1438-1465`).
#[test]
fn a_pool_doubted_on_every_worker_pairs_by_load() {
    let mut suspect = [doubted(0, 2), doubted(1, 1), doubted(2, 0)];
    suspect[2].health.breaker_open = true;
    let suspect = suspect.map(|worker| classed(worker, 1));
    let idle = [classed(free(0, 0), 1)];
    let pools = [
        (&suspect[..], &idle[..], (1, 0)),
        (&idle[..], &suspect[..], (0, 1)),
    ];
    for shed in [false, true] {
        for (prefills, decodes, pair) in pools {
            let paired = decide_pair(&pair_engine(shed), prefills, decodes);
            assert_eq!(pair_of(&paired), pair, "shedding: {shed}");
            assert_eq!(paired.trace().lifted, LiftedVeto::Liveness);
        }
    }
}

/// The admission limit has the last word on a pair's default. A free
/// prefill at its limit keeps the limit's refusal beside a prefill a
/// veto refuses: the request waits, and no veto is lifted for it. Where
/// the default takes a prefill at its limit the request is shed under
/// the limit too, and the next prefill is not taken in its place. The
/// gateway's own selection answers both so
/// (`model_gateway/src/routers/common/placement.rs:1467-1480`,
/// `:1148-1153`). Each of these sheds names the prefill side: that
/// selection refuses for the prefill's role at the limit (`:2980-2983`,
/// `:3048-3051`, on main `d43c7b2a9`).
#[test]
fn the_admission_limit_has_the_last_word_on_a_pairs_default() {
    let mut filters = vetoes();
    filters.push(Box::new(HasCapacity));
    let engine = pair_engine_of(false, [filters, vetoes()]);
    let room = |mut worker: Candidate| {
        worker.has_capacity = true;
        classed(worker, 1)
    };
    let decodes = [classed(free(0, 0), 1)];
    let full = (
        (ShedReason::NoEndpoints, "has-capacity"),
        Some(PairSide::Prefill),
    );
    for vetoed in [over(1, 0), doubted(1, 0)] {
        let prefills = [classed(free(0, 4), 1), room(vetoed)];
        assert_eq!(sided(&decide_pair(&engine, &prefills, &decodes)), full);
    }
    // So it does where every decode is over its thresholds and the rule
    // opened their class: the free prefill stands at its limit with an
    // open partner, and the busy prefill with a slot is not taken. The
    // gateway judges the limit over the prefills with an open partner
    // (`:2515-2528` there on main `7526c6664e`).
    let busy = [classed(over(0, 0), 1)];
    let prefills = [classed(free(0, 4), 1), room(over(1, 0))];
    assert_eq!(sided(&decide_pair(&engine, &prefills, &busy)), full);
    // Two prefills over their thresholds: the less loaded is taken, at
    // its limit and then with a slot.
    let busier = room(over(1, 5));
    let limited = decide_pair(&engine, &[classed(over(0, 4), 1), busier], &decodes);
    assert_eq!(sided(&limited), full);
    // That refusal names no lifted veto: no request was steered.
    assert_eq!(limited.trace().lifted, LiftedVeto::None);
    let paired = decide_pair(&engine, &[room(over(0, 4)), busier], &decodes);
    assert_eq!(pair_of(&paired), (0, 0));
    // The limit is set aside on the prefill side alone. A decode whose
    // own profile lists the limit, and that is at it, pairs with nobody:
    // the prefill over its thresholds has no decode to be steered to,
    // and the shed says so.
    let mut both = [vetoes(), vetoes()];
    for side in &mut both {
        side.push(Box::new(HasCapacity));
    }
    let both_limited = pair_engine_of(false, both);
    let alone = [room(over(0, 4))];
    let refused = decide_pair(&both_limited, &alone, &decodes);
    assert_eq!(shed_of(&refused), (ShedReason::NoEndpoints, PARTNERED));
    // At its own limit too that prefill names no shed under the limit:
    // no decode stands behind it, and a host that waits on the limit
    // would wait for nothing.
    let at_limit = [classed(over(0, 4), 1)];
    let refused = decide_pair(&both_limited, &at_limit, &decodes);
    assert_eq!(shed_of(&refused), (ShedReason::NoEndpoints, PARTNERED));
}

/// One rule names a pair's shed with no endpoint where the overload
/// veto ended the pass ([`Half::refused_room`]): the stage that refused
/// the worker with room the count ends at. The prefill's one partner
/// is over its thresholds, beside two decodes with room that pair with
/// nobody: one under a liveness doubt, and one that stands. The first
/// of the two in the list's order names the shed. A worker with room
/// that two stages refuse, the health stage and a filter of the
/// profile's own, is named by the first of them in the profile's
/// order, whichever that is. The admission limit names no such shed, on
/// the decode side either: a decode with room that its own limit alone
/// refuses pairs with nobody that stands. The rule is asked behind a
/// pass the overload veto ended, and behind no other: a sole partner
/// that its own limit alone refuses ends the pass at the limit, and
/// that shed keeps the limit's name.
///
/// Where no worker of the list has room and each ready one is over its
/// thresholds, a list no request is shed over under a binding that
/// steers, the first ready one is asked the same way.
#[test]
fn a_pairs_shed_with_no_endpoint_is_named_by_the_first_worker_with_room() {
    let shedding = pair_engine_of(true, [vetoes(), vetoes()]);
    let prefills = [classed(free(0, 0), 1)];
    let said = |engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]| {
        shed_of(&decide_pair(engine, prefills, decodes))
    };
    let (doubt, apart) = (
        (ShedReason::NoEndpoints, "healthy"),
        (ShedReason::NoEndpoints, PARTNERED),
    );
    let partner = classed(over(0, 0), 1);
    let decodes = [partner, classed(doubted(1, 0), 2), classed(free(2, 0), 2)];
    assert_eq!(said(&shedding, &prefills, &decodes), doubt);
    let decodes = [partner, classed(free(1, 0), 2), classed(doubted(2, 0), 2)];
    assert_eq!(said(&shedding, &prefills, &decodes), apart);
    // The decode side lists its admission limit behind its two vetoes,
    // and the doubted decode is at it.
    let mut filters = vetoes();
    filters.push(Box::new(HasCapacity));
    let limited = pair_engine_of(true, [vetoes(), filters]);
    let decodes = [partner, classed(doubted(1, 0), 2)];
    assert_eq!(said(&limited, &prefills, &decodes), doubt);
    // A decode with room that its own limit alone refuses: the limit
    // is set aside for the name, since a host waits on that name for a
    // prefill's room, and no prefill waits here.
    let decodes = [partner, classed(free(1, 0), 2)];
    assert_eq!(said(&limited, &prefills, &decodes), apart);
    // So it is where the decode side lists its limit ahead of its two
    // vetoes or between them, and the partner over its thresholds has
    // a slot: the rule sets the limit aside wherever the profile lists
    // it.
    let mut slotted = partner;
    slotted.has_capacity = true;
    for at in [0, 1] {
        let mut listed = vetoes();
        listed.insert(at, Box::new(HasCapacity));
        let limit_at = pair_engine_of(true, [vetoes(), listed]);
        let decodes = [slotted, classed(free(1, 0), 2)];
        assert_eq!(said(&limit_at, &prefills, &decodes), apart, "{at}");
    }
    // The rule is asked behind a pass that the overload veto ended, and
    // behind no other. A sole partner under its thresholds that its own
    // limit alone refuses ends the partners' pass at the limit: that
    // shed keeps the limit's name, for the decode side.
    let full = (ShedReason::NoEndpoints, "has-capacity");
    let decodes = [classed(free(0, 0), 1)];
    let at_limit = decide_pair(&limited, &prefills, &decodes);
    assert_eq!(sided(&at_limit), (full, Some(PairSide::Decode)));
    // A decode with room that two stages refuse: the health stage, and
    // a filter that refuses every worker. The first of the two in the
    // profile's order names the shed: the health stage where the filter
    // is listed behind the two vetoes, and the filter where it is
    // listed ahead of the health stage.
    let mut twice = vetoes();
    twice.push(Box::new(Refuses(Arc::default())));
    let refused_twice = pair_engine_of(true, [vetoes(), twice]);
    let decodes = [partner, classed(doubted(1, 0), 2)];
    assert_eq!(said(&refused_twice, &prefills, &decodes), doubt);
    let turned: Vec<Box<dyn Filter>> = vec![
        Box::new(NotOverloaded),
        Box::new(Refuses(Arc::default())),
        Box::new(Healthy),
    ];
    let refuses_first = pair_engine_of(true, [vetoes(), turned]);
    // The shed the rule names is the decode side's: a prefill that
    // pairs stands.
    let refused = (ShedReason::NoEndpoints, "refuses");
    let named = decide_pair(&refuses_first, &prefills, &decodes);
    assert_eq!(sided(&named), (refused, Some(PairSide::Decode)));
    // Two prefills over their thresholds, one of them under a doubt,
    // beside a decode that is down: no pair to be steered to.
    let steering = pair_engine_of(false, [vetoes(), vetoes()]);
    let mut down = classed(free(0, 0), 1);
    down.health.healthy = false;
    let prefills = [classed(doubted_over(0, 0), 1), classed(over(1, 0), 1)];
    assert_eq!(said(&steering, &prefills, &[down]), doubt);
    let prefills = [classed(over(0, 0), 1), classed(doubted_over(1, 0), 1)];
    assert_eq!(said(&steering, &prefills, &[down]), apart);
}

/// The pair `engine` decides over [`first_pools`].
fn decide_first(engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]) -> Decision {
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let pools = first_pools(prefills, decodes);
    engine.decide_pair(&facts(1), &pool, pools, &cx, &cx, &mut Scratch::new())
}

/// While a prefill has a first partner open, the pair is made among
/// such prefills and their first partners, whatever stands beside
/// them, and a request waits where the admission limit alone holds
/// such a prefill back. A prefill another stage refuses keeps no
/// request: the pair is then made among every prefill with an open
/// partner, as it is where no first partner is open. The gateway's own
/// selection orders its candidates so, asks the limit of them, and
/// takes the pick's decode among its partners of that order
/// (`model_gateway/src/routers/common/placement.rs:2479-2502`,
/// `:2532-2545`, `:2615-2618` on main `30f636706`).
#[test]
fn a_pair_is_made_among_first_partners_while_one_is_open() {
    let mut filters = vetoes();
    filters.push(Box::new(HasCapacity));
    let engine = pair_engine_of(false, [filters, vetoes()]);
    let room = |mut worker: Candidate, class: u32| {
        worker.has_capacity = true;
        classed(worker, class)
    };
    // Place 0 of each side holds a worker with no first partner, and
    // place 1 the two workers that pair first.
    let decodes = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    let prefills = [room(free(0, 0), 2), room(free(1, 0), 1)];
    assert_eq!(pair_of(&decide_first(&engine, &prefills, &decodes)), (1, 1));
    // With no first partner stated the same lists pair by the picker
    // alone, which takes the other prefill.
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    let (pools, scratch) = (first_pools(&prefills, &decodes), &mut Scratch::new());
    let plain = PairPools {
        first: FirstPartners::default(),
        ..pools
    };
    let unordered = engine.decide_pair(&req, &pool, plain, &cx, &cx, scratch);
    assert_eq!(pair_of(&unordered), (0, 1));
    // Each side's shadow is shown what its live side was shown: the
    // prefills with a first partner open, and that pair's first
    // partners. The decodes stand the other way round here, so the
    // picker alone takes neither worker of the pair.
    let turned = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
    let pools = first_pools(&prefills, &turned);
    let live = engine.decide_pair(&req, &pool, pools, &cx, &cx, scratch);
    assert_eq!(pair_of(&live), (1, 0));
    let pick = live.pick().expect("a pair");
    for (side, shown) in [(PairSide::Prefill, 1), (PairSide::Decode, 0)] {
        let twin = engine.shadow_side(
            0,
            side,
            &req,
            &pool,
            pools,
            &cx,
            &cx,
            pick,
            scratch,
            &|_| None,
        );
        assert_eq!(
            twin.map(|shadow| order(&shadow)[0]),
            Some(shown),
            "{side:?}"
        );
    }
    // Under no table a class pairs with its own number, and no first
    // partner is read, whatever an earlier decision marked in the
    // buffer.
    let bare = PairPools {
        table: PairTable::default(),
        ..first_pools(&prefills, &decodes)
    };
    let numbered = engine.decide_pair(&req, &pool, bare, &cx, &cx, scratch);
    assert_eq!(pair_of(&numbered), (0, 0));
    // At its admission limit the prefill with a first partner open
    // keeps the request: it waits, beside a prefill with a slot and an
    // open partner.
    let full = [room(free(0, 0), 2), classed(free(1, 0), 1)];
    let limit = (ShedReason::NoEndpoints, "has-capacity");
    assert_eq!(
        sided(&decide_first(&engine, &full, &decodes)),
        (limit, Some(PairSide::Prefill))
    );
    // Refused by another stage it keeps none, with a slot or at its
    // limit: the pair is made among every prefill with an open partner.
    let mut down = free(1, 0);
    down.health.healthy = false;
    for refused in [over(1, 0), down] {
        for prefill in [room(refused, 1), classed(refused, 1)] {
            let next = decide_first(&engine, &[room(free(0, 0), 2), prefill], &decodes);
            assert_eq!(pair_of(&next).0, 0);
        }
    }
    // No first partner is open: a prefill pairs with any open partner
    // of its, the one with first partners too.
    let closed = [classed(free(0, 0), 2), classed(over(1, 0), 1)];
    let alone = [classed(free(0, 0), 2), room(free(1, 0), 1)];
    assert_eq!(pair_of(&decide_first(&engine, &alone, &closed)), (1, 0));
}

/// The decode of a pair made among first partners is one of them under
/// the binding's all-vetoed rule too: with every decode over its
/// thresholds the rule opens both classes, and the first partner is
/// taken though the other decode holds fewer requests
/// (`model_gateway/src/routers/common/placement.rs:2454-2456`,
/// `:2615-2618` on main `30f636706`). The pair's default reads no
/// first partner: over a prefill pool refused whole it takes the pair
/// of the least load among every partner, as the gateway's default
/// does (`:1804-1813` there).
#[test]
fn a_first_pairs_decode_is_a_first_partner_under_the_rule() {
    let engine = pair_engine(false);
    let prefills = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    let busy = [classed(over(0, 0), 2), classed(over(1, 5), 1)];
    let steered = decide_first(&engine, &prefills, &busy);
    assert_eq!(pair_of(&steered), (1, 1));
    assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
    // The default: the prefill of the least load with the partner of
    // the fewest requests, which is no first partner of it.
    let refused = [classed(over(0, 5), 2), classed(over(1, 0), 1)];
    let decodes = [classed(free(0, 0), 2), classed(free(1, 3), 1)];
    let paired = decide_first(&engine, &refused, &decodes);
    assert_eq!(pair_of(&paired), (1, 0));
    assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
}

/// A decode asked again behind its prefill is put to the test a pair's
/// first decision makes, at its own instant: it is one of the held
/// prefill's first partners while one of them is open, or opens where
/// the rule lifts the overload veto for partners that are all over
/// their thresholds, and any partner of the prefill else. The gateway's
/// own selection asks no decode again, so these answers are the
/// engine's own.
#[test]
fn a_decode_asked_again_is_a_first_partner_while_one_is_open() {
    let engine = pair_engine(false);
    let prefills = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    let asked = |held: usize, decodes: &[Candidate]| {
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let (pools, id) = (first_pools(&prefills, decodes), prefills[held].id);
        engine.decide_decode(&facts(1), &pool, pools, id, None, &cx, &mut Scratch::new())
    };
    let open = [classed(free(0, 0), 2), classed(free(1, 0), 1)];
    assert_eq!(pair_of(&asked(1, &open)), (1, 1));
    // The first partner over its thresholds beside an open partner:
    // the open one, with no veto lifted.
    let busy = [classed(free(0, 0), 2), classed(over(1, 0), 1)];
    let other = asked(1, &busy);
    assert_eq!(pair_of(&other), (1, 0));
    assert_eq!(other.trace().lifted, LiftedVeto::None);
    // Every partner over its thresholds: the rule's lift opens the
    // first partner, which holds more requests than the other.
    let over_all = [classed(over(0, 0), 2), classed(over(1, 5), 1)];
    let steered = asked(1, &over_all);
    assert_eq!(pair_of(&steered), (1, 1));
    assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
    // A first partner the lifted veto does not open leaves the rule
    // every partner.
    let mut down = over(1, 5);
    down.health.healthy = false;
    let steered = asked(1, &[classed(over(0, 0), 2), classed(down, 1)]);
    assert_eq!(pair_of(&steered), (1, 0));
    // The prefill with no first partner is asked among all its
    // partners: the fewest requests under the rule.
    let steered = asked(0, &over_all);
    assert_eq!(pair_of(&steered), (0, 0));
    assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
    // The shadow of a decode asked again reads the partners that call
    // read, the held prefill's first partners. It is asked with the
    // decodes both ways round: the picker alone takes one place of the
    // two, whichever class stands there.
    let turned = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
    let (req, pool, cx) = (facts(1), PoolFacts::default(), Constraints::default());
    for (decodes, first) in [(&open, 1), (&turned, 0)] {
        let (pools, held) = (first_pools(&prefills, decodes), prefills[1].id);
        let scratch = &mut Scratch::new();
        let again = engine.decide_decode(&req, &pool, pools, held, None, &cx, scratch);
        assert_eq!(pair_of(&again), (1, first));
        let later =
            engine.shadow_decode_later(0, &req, &pool, pools, held, None, &cx, scratch, &|_| None);
        assert_eq!(later.map(|shadow| order(&shadow)[0]), Some(first));
    }
}

/// The pools of a first list that is no partner list of the table: the
/// table pairs class 1 with class 1 alone, and the first list names
/// class 2.
#[cfg(debug_assertions)]
fn stray_pools<'a>(prefills: &'a [Candidate], decodes: &'a [Candidate]) -> PairPools<'a> {
    PairPools {
        prefills,
        decodes,
        table: PairTable::new(&[1], &[1]),
        first: FirstPartners::new(&[1], &[2]),
        ..PairPools::default()
    }
}

/// First partners that do not stand within the pair table stop a debug
/// build where a decision takes them: at a pair's first decision.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the first partners do not stand within the pair table: Err(NoPartner)")]
fn first_partners_outside_the_table_stop_a_debug_build_at_a_pairs_decision() {
    let engine = pair_engine(false);
    let (prefills, decodes) = ([classed(free(0, 0), 1)], [classed(free(0, 0), 1)]);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let pools = stray_pools(&prefills, &decodes);
    let pair = engine.decide_pair(&facts(1), &pool, pools, &cx, &cx, &mut Scratch::new());
    panic!("first partners outside the table answered {pair:?}");
}

/// And at a decode asked again behind its prefill, which reads the held
/// prefill's first list as it is handed.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the first partners do not stand within the pair table: Err(NoPartner)")]
fn first_partners_outside_the_table_stop_a_debug_build_at_a_decode_asked_again() {
    let engine = pair_engine(false);
    let (prefills, decodes) = ([classed(free(0, 0), 1)], [classed(free(0, 0), 1)]);
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let (pools, held) = (stray_pools(&prefills, &decodes), prefills[0].id);
    let scratch = &mut Scratch::new();
    let again = engine.decide_decode(&facts(1), &pool, pools, held, None, &cx, scratch);
    panic!("first partners outside the table answered {again:?}");
}

/// The pair's default asks each side's hard filters alone
/// ([`Half::stands`]). The one decode has no room, so a soft filter of
/// its profile refuses it, and the default still takes it for the
/// prefill pool it answers for: a room steers no default, as no soft
/// filter does.
#[test]
fn a_pairs_default_asks_no_soft_filter() {
    let mut roomed = vetoes();
    roomed.push(Box::new(HasRoom));
    let engine = pair_engine_of(false, [vetoes(), roomed]);
    let paired = decide_pair(
        &engine,
        &[classed(over(0, 1), 1)],
        &[classed(free(0, 0), 1)],
    );
    assert_eq!(pair_of(&paired), (0, 0));
    assert_eq!(paired.trace().lifted, LiftedVeto::Overload);
}

/// Each side of the pair's default is steered by its own facts. Equal
/// pairs are drawn by the prefill side's seed, which is the request's,
/// and equal partners by the decode side's own draw sequence: seed by
/// seed, the pair is the worker the rule draws over each side's rows
/// alone under that side's seed. The pair's trace is the prefill's: it
/// is made under the prefill profile and carries the request's seed.
#[test]
fn a_pairs_default_steers_each_side_by_its_own_facts() {
    let (engine, alone) = (pair_engine(false), engine_of("round_robin", false));
    let prefills = [classed(over(0, 2), 1), classed(over(1, 2), 1)];
    let decodes = [classed(over(0, 3), 1), classed(over(1, 3), 1)];
    let first = |seed: u64, pool: &[Candidate]| order(&decide(&alone, &facts(seed), pool))[0];
    let mut parted = 0;
    for seed in 1..=64 {
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let scratch = &mut Scratch::new();
        let paired = engine.decide_pair(&facts(seed), &pool, pools, &cx, &cx, scratch);
        let own = crate::Draw::new(seed).next_u64();
        let drawn = (first(seed, &prefills), first(own, &decodes));
        assert_eq!(pair_of(&paired), drawn, "seed {seed}");
        assert_eq!(paired.trace().seed, seed);
        assert_eq!(paired.trace().made_under(), Some(0));
        parted += u32::from(first(seed, &decodes) != drawn.1);
    }
    // The decode side's sequence is no copy of the request's.
    assert!(parted > 0, "the two sides drew alike under every seed");
}

/// A free decode keeps its pool from the rule, whoever it pairs with: a
/// decode outside any pairing, or one whose own prefill is not ready.
/// The gateway's own pair selection lifts the veto for the decode side
/// only where no decode of the whole pool is available
/// (`model_gateway/src/routers/common/placement.rs:1315-1321`, `:1341`),
/// and refuses these pairs. Its default answers for no such pool either:
/// the decode pool is not vetoed on every worker (`:1094`, `:1443-1448`).
/// It answers that it has no worker, and the decision sheds with no
/// endpoint: the free decode pairs with nobody that stands.
#[test]
fn a_free_decode_keeps_its_pool_from_the_rule() {
    let engine = pair_engine(false);
    let refused = |prefills: &[Candidate], decodes: &[Candidate]| {
        shed_of(&decide_pair(&engine, prefills, decodes))
    };
    let apart = (ShedReason::NoEndpoints, PARTNERED);
    let outside = [classed(over(0, 0), 1), free(1, 0)];
    assert_eq!(refused(&[classed(free(0, 0), 1)], &outside), apart);
    let mut prefills = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
    prefills[1].health.healthy = false;
    let decodes = [classed(over(0, 0), 1), classed(free(1, 0), 2)];
    assert_eq!(refused(&prefills, &decodes), apart);
}

/// A room narrows a prefill's partners, and hides no prefill. Both
/// decodes are over their thresholds alone: one is full and its prefill
/// free, the other has room and its prefill is at its limit. The pair is
/// the free prefill with its full decode, whichever of the two classes
/// stands first among the decodes. Between two partners of one prefill
/// the one with a room takes the pair. The gateway's own pair
/// selection makes it so: a prefill at its limit leaves before the
/// policy picks, and the rooms are asked among the picked prefill's
/// partners (`model_gateway/src/routers/common/placement.rs:1467-1480`,
/// `:1520-1521`, `:1551-1578`).
#[test]
fn a_room_narrows_a_prefills_partners_and_hides_no_prefill() {
    let (mut limit, mut room) = (vetoes(), vetoes());
    limit.push(Box::new(HasCapacity));
    room.push(Box::new(HasRoom));
    let engine = pair_engine_of(false, [limit, room]);
    // The pair where the prefill at `under` is under its limit and the
    // other prefill's decode has the room.
    let placed = |under: usize| {
        let mut prefills = [classed(free(0, 0), 1), classed(free(1, 0), 2)];
        prefills[under].has_capacity = true;
        let mut decodes = [classed(over(0, 0), 1), classed(over(1, 0), 2)];
        decodes[1 - under].has_room = true;
        let decision = decide_pair(&engine, &prefills, &decodes);
        (pair_of(&decision), decision.trace().lifted)
    };
    assert_eq!(placed(0), ((0, 0), LiftedVeto::Overload));
    assert_eq!(placed(1), ((1, 1), LiftedVeto::Overload));
    // The partner with a room, though the full one holds fewer requests.
    let mut prefill = classed(free(0, 0), 1);
    prefill.has_capacity = true;
    let mut partners = [classed(over(0, 0), 1), classed(over(1, 3), 1)];
    partners[1].has_room = true;
    let decision = decide_pair(&engine, &[prefill], &partners);
    assert_eq!(pair_of(&decision), (0, 1));
}

/// Three partners of class 1 over their thresholds, holding `in_flight`
/// requests in their order, and their ids. The ids are their own: a
/// tried list names a decode and no prefill.
fn partners(in_flight: [u32; 3]) -> ([Candidate; 3], [WorkerId; 3]) {
    let mut decodes = [0, 1, 2].map(|at| classed(over(at, in_flight[at as usize]), 1));
    for decode in &mut decodes {
        decode.id = WorkerId(200 + u64::from(decode.position));
    }
    (decodes, decodes.map(|decode| decode.id))
}

/// The decodes a request is given among `decodes`, one prefill's
/// partners, best first. It tried `tried`, and its host just failed to
/// reach `gone`. A first decision names them. A decode asked again
/// behind the prefill names the same over a scratch of its own, and
/// both say the lifted overload veto.
fn given(
    engine: &Engine,
    decodes: &[Candidate],
    tried: &[WorkerId],
    gone: &[WorkerId],
) -> Vec<u32> {
    let prefills = [classed(free(0, 0), 1)];
    let pools = PairPools {
        prefills: &prefills,
        decodes,
        ..PairPools::default()
    };
    // A request that names a tried worker says its retry number.
    let req = RequestFacts {
        tried,
        attempt: u8::from(!tried.is_empty()),
        ..facts(1)
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let reached = Constraints {
        excluded: gone,
        ..Constraints::default()
    };
    let first = engine.decide_pair(&req, &pool, pools, &cx, &reached, &mut Scratch::new());
    let (held, fresh) = (prefills[0].id, &mut Scratch::new());
    let again = engine.decide_decode(&req, &pool, pools, held, None, &reached, fresh);
    let named = |decision: &Decision| {
        assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
        let targets = decision.pick().expect("a pick").targets.iter();
        let decodes = targets.map(|target| target.pair.expect("a pair").decode_position);
        decodes.collect::<Vec<_>>()
    };
    assert_eq!(named(&first), named(&again));
    named(&first)
}

/// The pass that opens the prefills sets the retry's tried list aside,
/// and the partners' pass does where reading the list leaves nobody: a
/// hard filter that reads the list hides no prefill and refuses no lone
/// partner. The gateway's own pair selection steps aside from a tried
/// decode only while another partner is left
/// (`model_gateway/src/routers/common/placement.rs:1562-1570`), and
/// places this pair.
///
/// The partners' pass sets the list aside too where the profile's
/// `not-tried` stood aside in it. Behind the hard filter that filter
/// keeps a partner the host just failed to reach alone, and the request
/// goes to the tried partners the host can reach. Under a profile that
/// lists no `not-tried` no filter stood aside and none reads the host's
/// list: the hard filter's pass stands as it keeps, and no tried
/// partner comes back beside an unreachable one. The gateway keeps one
/// list for a tried worker and one it failed to reach (`was_tried`,
/// `:1177-1179` there) and takes the partner outside it (`:1568-1570`).
#[test]
fn a_tried_decode_still_opens_its_prefill() {
    /// Refuses a worker an earlier attempt tried: a hard filter.
    #[derive(Debug)]
    struct Untried;
    impl Filter for Untried {
        fn name(&self) -> &'static str {
            "untried"
        }
        fn keep(&self, req: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
            !req.tried.contains(&row.id)
        }
    }
    let mut reads_tried = vetoes();
    reads_tried.push(Box::new(Untried));
    let engine = pair_engine_of(false, [vetoes(), reads_tried]);
    let (prefills, mut decodes) = ([classed(free(0, 0), 1)], [classed(over(0, 1), 1)]);
    // An id of its own: the tried list names the decode and no prefill.
    decodes[0].id = WorkerId(200);
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let tried = [decodes[0].id];
    let req = RequestFacts {
        tried: &tried,
        ..facts(1)
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let decision = engine.decide_pair(&req, &pool, pools, &cx, &cx, &mut Scratch::new());
    assert_eq!(pair_of(&decision), (0, 0));
    assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
    // The pair's default reads the request's own facts. The same
    // partner, free and tried, stays refused by the hard filter as on
    // main: no veto refuses it, so none is lifted for it.
    let mut idle = decodes;
    idle[0].health.overloaded = false;
    let pools = PairPools {
        decodes: &idle,
        ..pools
    };
    let decision = engine.decide_pair(&req, &pool, pools, &cx, &cx, &mut Scratch::new());
    assert_eq!(shed_of(&decision), (ShedReason::NoEndpoints, "untried"));
    // Two tried partners beside one the host just failed to reach,
    // under `not-tried` behind the hard filter: the tried ones.
    let (decodes, ids) = partners([0, 1, 2]);
    let mut listed = vetoes();
    listed.push(Box::new(Untried));
    listed.push(Box::new(NotTried));
    let listed = pair_engine_of(false, [vetoes(), listed]);
    assert_eq!(given(&listed, &decodes, &ids[..2], &ids[2..]), [0, 1]);
    // Under the hard filter alone its pass stands on the two partners
    // it keeps. The host just failed to reach the first, which is the
    // tried one, and the third: the second takes the request, as in
    // the gateway, and the first does not come back.
    let gone = [ids[0], ids[2]];
    assert_eq!(given(&engine, &decodes, &ids[..1], &gone), [1, 2]);
    // That profile reads no host list: an unreachable second partner
    // stays ahead of the third by its load, where the gateway's one
    // list leaves the third alone.
    assert_eq!(given(&engine, &decodes, &ids[..1], &ids[1..2]), [1, 2]);
}

/// A retry over partners all over their thresholds keeps to the ones the
/// request has not tried while one is left. The fewest in flight among
/// those takes it, and the next fewest of them stand behind it.
/// With every partner tried the list narrows nobody. A tried partner
/// with a room stays ahead of untried ones with none. The gateway's own
/// pair selection orders a pick's partners so: the rooms, the untried
/// ones, the least loaded
/// (`model_gateway/src/routers/common/placement.rs:1561`, `:1568-1570`,
/// `:1575-1578`, `narrow_to` at `:266-282`). It names one pair
/// (`:1621-1625`), so the alternates are the engine's own. Where the
/// list names every partner but one the host just failed to reach, that
/// one stays the last resort: the gateway's retry keeps one list, so
/// that answer is the engine's own. So is a decode asked again behind
/// its prefill: the gateway asks no decode again. So is the answer
/// under a document that lists `not-tried` ahead of the room filter,
/// an order the gateway has not: a partner the host just failed to
/// reach takes no retry there from tried ones the host can reach.
#[test]
fn a_retry_keeps_to_the_untried_partners_while_one_is_left() {
    // The decode side's filters in the pair presets' order.
    let mut reads_tried = vetoes();
    reads_tried.push(Box::new(HasRoom));
    reads_tried.push(Box::new(NotTried));
    let engine = pair_engine_of(false, [vetoes(), reads_tried]);
    // The lightest untried partner stands last among them.
    let (decodes, ids) = partners([0, 2, 1]);
    let mut roomy = decodes;
    roomy[0].has_room = true;
    // Where the list moves no answer: none, every partner tried, every
    // partner but one the host failed to reach, a tried one with a room.
    assert_eq!(given(&engine, &decodes, &[], &[]), [0, 2, 1]);
    assert_eq!(given(&engine, &decodes, &ids, &[]), [0, 2, 1]);
    assert_eq!(given(&engine, &decodes, &ids[1..], &ids[..1]), [2, 1]);
    assert_eq!(given(&engine, &roomy, &ids[..1], &[]), [0]);
    // Where it does: the untried partners while one is left, the
    // fewest in flight first, and no tried one behind them.
    assert_eq!(given(&engine, &decodes, &ids[..1], &[]), [2, 1]);
    assert_eq!(given(&engine, &decodes, &ids[..2], &[]), [2]);
    // A document may list `not-tried` ahead of the room filter. The
    // one partner with a room is one the host just failed to reach,
    // and the others are tried: the pass that reads the list keeps
    // that one alone, and the tried ones take the retry. In the
    // presets' order the room goes first, as in the gateway.
    let mut reversed = vetoes();
    reversed.push(Box::new(NotTried));
    reversed.push(Box::new(HasRoom));
    let reversed = pair_engine_of(false, [vetoes(), reversed]);
    assert_eq!(given(&reversed, &roomy, &ids[1..], &ids[..1]), [2, 1]);
    assert_eq!(given(&engine, &roomy, &ids[1..], &ids[..1]), [0]);
}

/// Under a profile that lists no `not-tried`, a filter of the host's
/// that reads the tried list answers on the request's own facts in the
/// partners' pass. One that keeps the tried partner alone leaves the
/// retry on it, however busy, and the pass with the list set aside is
/// not run. The gateway's own selection steps aside from a tried worker
/// while another is left
/// (`model_gateway/src/routers/common/placement.rs:1485-1494`,
/// `:1562-1570`), and asks no filter of a host's: this answer is the
/// engine's own.
#[test]
fn a_hosts_filter_that_keeps_a_tried_partner_stands_on_a_retry() {
    /// Keeps the workers an earlier attempt tried, and every worker of
    /// a request that tried none: a hard filter.
    #[derive(Debug)]
    struct Again;
    impl Filter for Again {
        fn name(&self) -> &'static str {
            "again"
        }
        fn keep(&self, req: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
            req.tried.is_empty() || req.tried.contains(&row.id)
        }
    }
    let mut sticky = vetoes();
    sticky.push(Box::new(Again));
    let engine = pair_engine_of(false, [vetoes(), sticky]);
    let (decodes, ids) = partners([0, 1, 2]);
    assert_eq!(given(&engine, &decodes, &[], &[]), [0, 1, 2]);
    assert_eq!(given(&engine, &decodes, &ids[2..], &[]), [2]);
}

/// A decode asked for a held prefill is put to the rule, both its tiers.
/// Partners over their thresholds on every worker go to the one that
/// holds the fewest requests, and under a binding that sheds they are
/// shed as saturated. Partners doubted on every worker go to the fewest
/// requests under either word, and never to one a hold stands on. A
/// request that names no tried worker takes one pass under the lifted
/// overload veto: with no doubt among its partners, a filter behind the
/// veto is asked about each once, and its refusal stands. The gateway's
/// own selection asks no decode again, so these answers are the
/// engine's own.
#[test]
fn a_decode_asked_for_a_held_prefill_is_put_to_the_rule() {
    let prefills = [classed(free(0, 0), 1)];
    let asked = |engine: &Engine, decodes: &[Candidate]| {
        let pools = PairPools {
            prefills: &prefills,
            decodes,
            ..PairPools::default()
        };
        engine.decide_decode(
            &facts(1),
            &PoolFacts::default(),
            pools,
            prefills[0].id,
            None,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    let busy = [over(0, 2), over(1, 1)].map(|worker| classed(worker, 1));
    let steered = asked(&pair_engine(false), &busy);
    assert_eq!(pair_of(&steered), (0, 1));
    assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
    assert_eq!(
        shed_of(&asked(&pair_engine(true), &busy)),
        (ShedReason::Saturated, "not-overloaded")
    );
    // The request names no tried worker: one pass under the lifted veto.
    let count = Arc::new(AtomicUsize::new(0));
    let mut refused = vetoes();
    refused.push(Box::new(Refuses(Arc::clone(&count))));
    let shed = asked(&pair_engine_of(false, [vetoes(), refused]), &busy);
    assert_eq!(shed_of(&shed), (ShedReason::NoEndpoints, "refuses"));
    assert_eq!(count.load(Ordering::Relaxed), busy.len());
    let mut suspect = [doubted(0, 2), doubted(1, 1), doubted(2, 0)];
    suspect[2].health.breaker_open = true;
    let suspect = suspect.map(|worker| classed(worker, 1));
    for shed in [false, true] {
        let steered = asked(&pair_engine(shed), &suspect);
        assert_eq!(pair_of(&steered), (0, 1), "shedding: {shed}");
        assert_eq!(steered.trace().lifted, LiftedVeto::Liveness);
    }
}
