use super::*;

/// A pair's dispatch moves its key's pin to the worker of the side it
/// names: the pair's decode worker on the decode side, the target
/// itself on the prefill side.
#[test]
fn a_pairs_dispatch_pins_the_worker_of_its_side() {
    let engine = session_engine(SESSIONS);
    let target = Target {
        id: WorkerId(3),
        pair: Some(Pair {
            decode_id: WorkerId(5),
            ..Pair::default()
        }),
        ..Target::default()
    };
    let lease = |key| Lease {
        session: session(key),
        session_verdict: SessionVerdict::Pinned,
        ..Lease::default()
    };
    engine.on_dispatched(&lease(7), &target, Side::Decode, 0);
    engine.on_dispatched(&lease(8), &target, Side::Prefill, 0);
    let pin = |key| engine.sessions.pins(session(key)).map(|pins| pins[0]);
    assert_eq!((pin(7), pin(8)), (Some(WorkerId(5)), Some(WorkerId(3))));
}

/// `profiles` under the pair plan and a sessions rule with no cap that
/// places a new session by `assignment`: the first profile on the
/// prefill side and the last on the decode side.
fn pair_session_engine(profiles: Vec<Profile>, assignment: Assignment) -> Engine {
    let rule = SessionRule {
        assignment,
        inflight_cap: 0,
        ..SESSIONS
    };
    let binding = Binding {
        decode_profile: Some(profiles.len() as u16 - 1),
        sessions: Some(rule),
        ..pair_binding()
    };
    Engine::new(profiles, vec![binding], 1, Needs::default())
}

/// A pair's request, under the session key `key` where one is given.
fn pair_request(key: Option<u64>) -> RequestFacts<'static> {
    RequestFacts {
        session_key: key.map_or(Identity::NONE, Identity::hashed),
        ..request(7)
    }
}

/// Each side's own session of a pair's request: the prefill side's,
/// then the decode side's.
fn side_sessions(facts: &RequestFacts<'_>) -> [Session; 2] {
    [Phase::Prefill, Phase::Decode].map(|phase| RequestFacts { phase, ..*facts }.session())
}

/// Prefills and decodes that pair all with all, each side under ids of
/// its own and the loads given: the prefills, then the decodes.
fn two_sides(prefill_loads: &[u32], decode_loads: &[u32]) -> [Vec<Candidate>; 2] {
    let side = |loads: &[u32], first: u64| -> Vec<Candidate> {
        let mut rows = classed_rows(&vec![1; loads.len()]);
        for (row, &load) in rows.iter_mut().zip(loads) {
            (row.id, row.in_flight) = (WorkerId(first + u64::from(row.position)), load);
        }
        rows
    };
    [side(prefill_loads, 100), side(decode_loads, 200)]
}

/// One pair decided for `facts` and dispatched a side at a time, as a
/// host commits it: each side's lease states that side's session and
/// its verdict, the decode side's as the first target's pair keeps it.
/// The two places, then the two verdicts.
fn pair_placed(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    [prefills, decodes]: &[Vec<Candidate>; 2],
    scratch: &mut Scratch,
) -> ((u32, u32), (SessionVerdict, DecodeVerdict)) {
    let pools = PairPools {
        prefills,
        decodes,
        ..PairPools::default()
    };
    pair_placed_in(engine, facts, pools, scratch)
}

/// [`pair_placed`] over the pools as a host hands them, with its pair
/// table and its first partners.
fn pair_placed_in(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    pools: PairPools<'_>,
    scratch: &mut Scratch,
) -> ((u32, u32), (SessionVerdict, DecodeVerdict)) {
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let decision = engine.decide_pair(facts, &pool, pools, &cx, &cx, scratch);
    let pick = decision.pick().expect("a pair");
    let target = *pick.targets.first().expect("a target");
    let pair = target.pair.expect("a decode");
    let [prefill_session, decode_session] = side_sessions(facts);
    let prefill = Lease {
        session: prefill_session,
        session_verdict: pick.session_verdict,
        ..Lease::default()
    };
    let went = Target {
        pair: None,
        ..target
    };
    engine.on_dispatched(&prefill, &went, Side::Prefill, 0);
    let decode = Lease {
        session: decode_session,
        session_verdict: pair.decode_verdict.on(pair.decode_id),
        ..Lease::default()
    };
    let went = Target {
        position: pair.decode_position,
        id: pair.decode_id,
        dp_rank: pair.decode_dp_rank,
        pair: None,
    };
    engine.on_dispatched(&decode, &went, Side::Decode, 0);
    (
        (target.position, pair.decode_position),
        (pick.session_verdict, pair.decode_verdict),
    )
}

/// Under a sessions rule each side of a pair follows its key's pin by
/// the side's own session. A key's first pair claims a worker on each
/// side: the pick states the prefill side's claim and the first
/// target's pair the decode side's. Each side's dispatch settles its
/// own claim, so the key's next pair names the same two workers as
/// pins, wherever the loads point by then. A request with no key is
/// placed as a pair under no rule is, states no verdict and pins
/// nothing.
///
/// The classic manual policy keeps one key's entry apart under each
/// role of a pair (`select_worker_impl` in
/// `model_gateway/src/policies/manual.rs`).
#[test]
fn each_side_of_a_pair_follows_its_keys_pin_by_its_own_session() {
    let engine = pair_session_engine(vec![least_requests_profile()], Assignment::Delegate);
    let mut scratch = Scratch::new();
    // The least loaded prefill is the second, the least loaded decode
    // the third.
    let first = two_sides(&[5, 0, 5], &[5, 5, 0]);
    let keyed = pair_request(Some(7));
    let (places, verdicts) = pair_placed(&engine, &keyed, &first, &mut scratch);
    assert_eq!(places, (1, 2));
    let claimed = SessionVerdict::Claimed(first[0][1].id);
    assert_eq!(verdicts, (claimed, DecodeVerdict::CLAIMED));
    let [prefill_session, decode_session] = side_sessions(&keyed);
    assert_ne!(prefill_session, decode_session);
    let pinned = |session| engine.sessions.pins(session).map(|pins| pins[0]);
    assert_eq!(pinned(prefill_session), Some(first[0][1].id));
    assert_eq!(pinned(decode_session), Some(first[1][2].id));
    // The loads point elsewhere on both sides: the two pins hold.
    let moved = two_sides(&[0, 9, 9], &[0, 9, 9]);
    let (places, verdicts) = pair_placed(&engine, &keyed, &moved, &mut scratch);
    assert_eq!(places, (1, 2));
    assert_eq!(verdicts, (SessionVerdict::Pinned, DecodeVerdict::PINNED));
    // No key: the pair a binding under no rule gives, and no verdict.
    let plain = pair_request(None);
    let (places, verdicts) = pair_placed(&engine, &plain, &moved, &mut scratch);
    assert_eq!(places, (0, 0));
    assert_eq!(verdicts, (SessionVerdict::None, DecodeVerdict::NONE));
    let unruled = pair_engine(least_requests_profile());
    let (same, _) = pair_placed(&unruled, &plain, &moved, &mut Scratch::new());
    assert_eq!(same, places);
    // A key under no rule is placed as a request with none is.
    let (same, none) = pair_placed(&unruled, &keyed, &moved, &mut Scratch::new());
    assert_eq!(same, places);
    assert_eq!(none, (SessionVerdict::None, DecodeVerdict::NONE));
    // The rule's assignment places a new session alone. A request with
    // no key is the profile's under any: here the idle worker of each
    // side holds the most keys, and takes it.
    let by_keys = pair_session_engine(vec![least_requests_profile()], Assignment::MinGroup);
    let mut keyed_rows = two_sides(&[0, 9, 9], &[0, 9, 9]);
    for side in &mut keyed_rows {
        side[0].active_sessions = 5;
    }
    let (same, none) = pair_placed(&by_keys, &plain, &keyed_rows, &mut Scratch::new());
    assert_eq!(same, places);
    assert_eq!(none, (SessionVerdict::None, DecodeVerdict::NONE));
    // A new key goes where the rule's assignment says: to the worker
    // of each side that holds the fewest keys, the last.
    for side in &mut keyed_rows {
        side[1].active_sessions = 1;
    }
    let (by_rule, claims) = pair_placed(&by_keys, &keyed, &keyed_rows, &mut Scratch::new());
    assert_eq!(by_rule, (2, 2));
    let claimed = SessionVerdict::Claimed(keyed_rows[0][2].id);
    assert_eq!(claims, (claimed, DecodeVerdict::CLAIMED));
}

/// A side whose picker leaves the placement to the rule is placed by
/// the rule's assignment, a request with a key and one with none
/// alike: under `min_load` each side takes its fewest in flight. The
/// key is pinned on each side and stays past the loads. The request
/// with no key follows the loads and pins nothing. Under no rule such
/// a side takes the fewest in flight too.
///
/// The classic manual policy places a request with no key by its
/// assignment under any role (`select_worker_impl` in
/// `model_gateway/src/policies/manual.rs`), and a new key the same way
/// (`select_by_routing_id`).
#[test]
fn a_manual_side_of_a_pair_is_placed_by_the_rules_assignment() {
    let engine = pair_session_engine(vec![manual_profile()], Assignment::MinLoad);
    let mut scratch = Scratch::new();
    let first = two_sides(&[3, 1, 2], &[2, 3, 1]);
    let keyed = pair_request(Some(7));
    let (places, verdicts) = pair_placed(&engine, &keyed, &first, &mut scratch);
    assert_eq!(places, (1, 2));
    let claimed = SessionVerdict::Claimed(first[0][1].id);
    assert_eq!(verdicts, (claimed, DecodeVerdict::CLAIMED));
    let moved = two_sides(&[0, 5, 2], &[1, 3, 9]);
    let (places, verdicts) = pair_placed(&engine, &keyed, &moved, &mut scratch);
    assert_eq!(places, (1, 2));
    assert_eq!(verdicts, (SessionVerdict::Pinned, DecodeVerdict::PINNED));
    let plain = pair_request(None);
    let (places, verdicts) = pair_placed(&engine, &plain, &moved, &mut scratch);
    assert_eq!(places, (0, 0));
    assert_eq!(verdicts, (SessionVerdict::None, DecodeVerdict::NONE));
    // Under no rule the picker's side reads the fewest in flight.
    let unruled = Engine::new(
        vec![manual_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    for facts in [&plain, &keyed] {
        let (places, verdicts) = pair_placed(&unruled, facts, &moved, &mut Scratch::new());
        assert_eq!(places, (0, 0));
        assert_eq!(verdicts, (SessionVerdict::None, DecodeVerdict::NONE));
    }
    // So it does on one side alone, beside a side whose picker picks:
    // on the decode side, then on the prefill side.
    for profiles in [
        vec![least_requests_profile(), manual_profile()],
        vec![manual_profile(), least_requests_profile()],
    ] {
        let one_side = Engine::new(
            profiles,
            vec![Binding {
                decode_profile: Some(1),
                ..pair_binding()
            }],
            1,
            Needs::default(),
        );
        let (places, _) = pair_placed(&one_side, &plain, &moved, &mut Scratch::new());
        assert_eq!(places, (0, 0));
    }
}

/// A prefill picker's decline passes through among first partners too:
/// the pair's decision is the decline, by the picker's name, and no
/// request waits for a prefill the picker was shown. It names the
/// prefill side: the gateway's own pair selection refuses for the
/// prefill's role where its prefill policy picks nobody
/// (`model_gateway/src/routers/common/placement.rs:3044`, on main
/// `d43c7b2a9`).
#[test]
fn a_prefill_pickers_decline_passes_through_among_first_partners() {
    let declining = Profile {
        name: "prefill".into(),
        filters: vec![Box::new(Healthy)],
        scorers: vec![],
        picker: Box::new(RefusesClass(1)),
    };
    let engine = pair_engine(declining);
    let mut prefill = candidate(0, 0);
    prefill.pair_class = 1;
    let mut decode = candidate(0, 0);
    decode.pair_class = 1;
    let (prefills, decodes) = ([prefill], [decode]);
    let first = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        table: PairTable::new(&[1], &[1]),
        first: FirstPartners::new(&[1], &[1]),
        ..PairPools::default()
    };
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let among = engine.decide_pair(&facts, &pool, first, &cx, &cx, &mut Scratch::new());
    for decision in [decide_pair(&engine, &prefills, &decodes), among] {
        match decision {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::PolicyDeclined);
                assert_eq!(shed.emptied_by, "refuses-class");
                assert_eq!(shed.side, Some(PairSide::Prefill));
            }
            other => panic!("a prefill picker that declines answered {other:?}"),
        }
    }
}

/// A rule places a pair's prefill side over the classes the side is
/// shown, in each of the two runs a host's first partners make.
///
/// Prefill class 1 pairs with both decode classes, and its first
/// partner is decode class 1. Prefill class 2 pairs with decode class
/// 2 and has no first partner. While no first partner is open, a new
/// key is placed over every open partner: on the idle prefill, of
/// class 2. Once the first partner opens, the side is shown the
/// prefills of class 1 alone. The key's pin stands outside them, so
/// the rule places the key again among them, and its decode is the
/// first partner. A pin's pick names the pin alone: shown every
/// class, the rule would follow the pin to a prefill with no first
/// partner, and the key would find no pair. Where the first run sheds,
/// the second places a new key over every open partner, by the rule
/// again: the pick states its claim. Each run leaves the classes it
/// was shown in the scratch, where a side's shadow reads them.
///
/// The classic router hands its policy the prefills that pair on a
/// known runtime alone (the candidates of `pair_pass` in
/// `model_gateway/src/routers/common/placement.rs`), and the classic
/// manual policy selects anew for a key whose workers are not among
/// those it is handed (`select_by_routing_id` in
/// `model_gateway/src/policies/manual.rs`).
#[test]
fn a_rule_places_a_pairs_prefill_among_first_partners_then_over_every_partner() {
    let engine = pair_session_engine(vec![least_requests_profile()], Assignment::Delegate);
    let mut scratch = Scratch::new();
    let table = PairTable::new(&[2, 3], &[1, 2, 2]);
    let first = FirstPartners::new(&[1, 1], &[1]);
    assert_eq!(first.within(&table), Ok(()));
    // A side's two workers, of class 1 and of class 2, under ids from
    // `first_id` on and the loads given.
    let rows = |loads: [u32; 2], first_id: u64| -> Vec<Candidate> {
        let mut rows = classed_rows(&[1, 2]);
        for (row, load) in rows.iter_mut().zip(loads) {
            (row.id, row.in_flight) = (WorkerId(first_id + u64::from(row.position)), load);
        }
        rows
    };
    // The idle worker of each side is the one of class 2.
    let (mut prefills, mut decodes) = (rows([5, 0], 100), rows([5, 0], 200));
    // The pair of `key`'s next request with its two verdicts, and the
    // classes its prefill side was shown.
    let placed = |key, prefills: &[Candidate], decodes: &[Candidate], scratch: &mut Scratch| {
        let pools = PairPools {
            prefills,
            decodes,
            table,
            first,
            ..PairPools::default()
        };
        let answer = pair_placed_in(&engine, &pair_request(Some(key)), pools, scratch);
        (answer, scratch.classes.clone())
    };
    let claim = DecodeVerdict::CLAIMED;
    let claimed = |id| (SessionVerdict::Claimed(WorkerId(id)), claim);
    // No first partner is open: decode class 1 has no healthy worker.
    decodes[0].health.healthy = false;
    let (answer, shown) = placed(7, &prefills, &decodes, &mut scratch);
    assert_eq!((answer, shown), (((1, 1), claimed(101)), vec![1, 2]));
    // The first partner opens, and the key's pin is not among the
    // prefills that have one.
    decodes[0].health.healthy = true;
    let (answer, shown) = placed(7, &prefills, &decodes, &mut scratch);
    assert_eq!((answer, shown), (((0, 0), claimed(100)), vec![1]));
    // The one prefill with a first partner open is not healthy: the
    // first run sheds, and the second is the rule's too.
    prefills[0].health.healthy = false;
    let (answer, shown) = placed(8, &prefills, &decodes, &mut scratch);
    assert_eq!((answer, shown), (((1, 1), claimed(101)), vec![1, 2]));
}

/// Where a host states no first partners, a rule's prefill run is
/// shown the prefills of the classes that have an open partner, as
/// the profile's run is. A key pinned to a prefill whose class lost
/// its last open partner stands outside them: the rule places the key
/// again among the prefills that pair, and a new key is not placed on
/// that prefill though it is the idle one. Shown every class, the
/// rule would follow the pin to a prefill no decode partners, and the
/// key would find no pair.
///
/// The classic router hands its policy the prefills with an open
/// partner alone (the candidates of `pair_pass` in
/// `model_gateway/src/routers/common/placement.rs`).
#[test]
fn a_rule_places_a_pairs_prefill_among_the_classes_with_an_open_partner() {
    let engine = pair_session_engine(vec![least_requests_profile()], Assignment::Delegate);
    let mut scratch = Scratch::new();
    // Each prefill class pairs with the decode class of its own number.
    let table = PairTable::new(&[1, 2], &[1, 2]);
    // A side's two workers, of class 1 and of class 2, under ids from
    // `first_id` on and the loads given.
    let rows = |loads: [u32; 2], first_id: u64| -> Vec<Candidate> {
        let mut rows = classed_rows(&[1, 2]);
        for (row, load) in rows.iter_mut().zip(loads) {
            (row.id, row.in_flight) = (WorkerId(first_id + u64::from(row.position)), load);
        }
        rows
    };
    // The idle worker of each side is the one of class 1.
    let (prefills, mut decodes) = (rows([0, 5], 100), rows([0, 5], 200));
    // The pair of `key`'s next request with its two verdicts, and the
    // classes its prefill side was shown.
    let placed = |key, decodes: &[Candidate], scratch: &mut Scratch| {
        let pools = PairPools {
            prefills: &prefills,
            decodes,
            table,
            ..PairPools::default()
        };
        let answer = pair_placed_in(&engine, &pair_request(Some(key)), pools, scratch);
        (answer, scratch.classes.clone())
    };
    let claim = DecodeVerdict::CLAIMED;
    let claimed = |id| (SessionVerdict::Claimed(WorkerId(id)), claim);
    // Both classes pair: the key is placed on the idle pair.
    let (answer, shown) = placed(7, &decodes, &mut scratch);
    assert_eq!((answer, shown), (((0, 0), claimed(100)), vec![1, 2]));
    // Class 1 loses its one decode: its prefill is shown no more, and
    // the key's pin stands on it.
    decodes[0].health.healthy = false;
    let (answer, shown) = placed(7, &decodes, &mut scratch);
    assert_eq!((answer, shown), (((1, 1), claimed(101)), vec![2]));
    // A new key is placed among the prefills that pair too.
    let (answer, shown) = placed(8, &decodes, &mut scratch);
    assert_eq!((answer, shown), (((1, 1), claimed(101)), vec![2]));
}

/// A side of a pair takes no turn of a list. Its request with no
/// session value never draws, at any pool size, so a new session of a
/// pair rotates through the survivors as that request does: on each
/// side the two kinds of request take one rotation, turn by turn.
///
/// The classic router asks one function for both kinds of request on a
/// side, and it steps the counter of the list it is handed
/// (`model_gateway/src/policies/round_robin.rs:142-167`).
#[test]
fn a_new_session_on_a_pairs_side_takes_the_survivors_rotation() {
    let rows = DRAW_FROM + 4;
    let engine = pair_session_engine(
        vec![round_robin_profile(), round_robin_profile()],
        Assignment::Delegate,
    );
    let sides = two_sides(&vec![0; rows], &vec![0; rows]);
    let mut scratch = Scratch::new();
    let keys = [None, Some(1), None, Some(2), Some(3), None];
    let places: Vec<(u32, u32)> = keys
        .into_iter()
        .map(|key| pair_placed(&engine, &pair_request(key), &sides, &mut scratch).0)
        .collect();
    let (prefill, decode) = places[0];
    let in_turn: Vec<(u32, u32)> = (0..keys.len() as u32)
        .map(|step| {
            (
                (prefill + step) % rows as u32,
                (decode + step) % rows as u32,
            )
        })
        .collect();
    assert_eq!(places, in_turn);
}

/// A decode profile whose picker declines the partners of a prefill
/// where the first of them is of pair class 7.
fn declining_decode_profile() -> Profile {
    Profile {
        name: "decode".into(),
        filters: vec![Box::new(Healthy)],
        scorers: vec![],
        picker: Box::new(RefusesClass(7)),
    }
}

/// A prefill the decode side declines is passed over under a rule too,
/// and the claim its side's run made moves with the pair. The pick
/// states the claim of the prefill that paired, and the key's pin names
/// that prefill before any dispatch. So a second decision of the key,
/// made before the first is dispatched, follows the prefill that pairs.
/// The first decision's dispatch there settles the claim.
#[test]
fn a_claim_on_a_prefill_that_is_passed_over_moves_to_the_one_that_pairs() {
    // The idle prefill's decode is refused; the busy prefill's is not.
    let mut sides = two_sides(&[0, 5], &[0, 0]);
    for side in &mut sides {
        (side[0].pair_class, side[1].pair_class) = (7, 8);
    }
    let profiles = vec![least_requests_profile(), declining_decode_profile()];
    let engine = pair_session_engine(profiles, Assignment::Delegate);
    let keyed = pair_request(Some(7));
    let [prefill_session, _] = side_sessions(&keyed);
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let paired = sides[0][1].id;
    // Two decisions of the key, neither dispatched: the first claims
    // the prefill that pairs, and the second follows that claim.
    let first = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    let first = first.pick().expect("a pair");
    let went = *first.targets.first().expect("a target");
    assert_eq!(went.id, paired);
    assert_eq!(first.session_verdict, SessionVerdict::Claimed(paired));
    let pinned = Some([paired, WorkerId(0)]);
    assert_eq!(engine.sessions.pins(prefill_session), pinned);
    let second = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    let second = second.pick().expect("the key's second pair");
    assert_eq!(second.targets.first().map(|target| target.id), Some(paired));
    assert_eq!(second.session_verdict, SessionVerdict::Pinned);
    // The first decision's dispatch on the prefill settles its claim.
    let lease = Lease {
        session: prefill_session,
        session_verdict: first.session_verdict,
        ..Lease::default()
    };
    let went = Target { pair: None, ..went };
    engine.on_dispatched(&lease, &went, Side::Prefill, 0);
    assert_eq!(engine.sessions.pins(prefill_session), pinned);
}

/// A pair's decision that forms no pair leaves no claim. Where the
/// decode side declines the partners of every prefill placed, the claim
/// the prefill side's run made is withdrawn, and the key holds no pin.
/// Its next request is placed again, and pairs with a prefill whose
/// decode is usable by then.
#[test]
fn a_pair_that_sheds_withdraws_the_claim_its_prefill_side_made() {
    // Every decode is of the refused class: no prefill pairs.
    let mut sides = two_sides(&[0, 5], &[0, 0]);
    for row in sides.iter_mut().flatten() {
        row.pair_class = 7;
    }
    let profiles = vec![least_requests_profile(), declining_decode_profile()];
    let engine = pair_session_engine(profiles, Assignment::Delegate);
    let keyed = pair_request(Some(7));
    let [prefill_session, _] = side_sessions(&keyed);
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let declined = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    assert!(declined.pick().is_none(), "{declined:?}");
    let none = Some([WorkerId(0); 2]);
    assert_eq!(engine.sessions.pins(prefill_session), none);
    // The busy prefill's decode is usable by the key's next request.
    for side in &mut sides {
        side[1].pair_class = 8;
    }
    let (places, verdicts) = pair_placed(&engine, &keyed, &sides, &mut scratch);
    assert_eq!(places, (1, 1));
    assert_eq!(verdicts.0, SessionVerdict::Claimed(sides[0][1].id));
}

/// A pair's release tells the session table both sides it ends ahead
/// of any stage: where a stage unwinds at the prefill's release under
/// a commit that refused every target, neither side keeps the worker
/// its decision claimed.
#[cfg(feature = "in-process")]
#[test]
fn a_stage_that_unwinds_at_a_pairs_release_leaves_no_claim_behind() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::{
        engine::{
            decision_engine::{HeldDecision, HeldDecode},
            lease_events,
        },
        tracking::lease::Said,
    };

    #[derive(Debug)]
    struct Unwinds;

    impl Lifecycle for Unwinds {
        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            panic!("this stage unwinds at a release");
        }
    }

    impl Scorer for Unwinds {
        fn name(&self) -> &'static str {
            "unwinds"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            for (score, _) in out.iter_mut().zip(kept) {
                *score = 1.0;
            }
            TieBand::NORMALIZED
        }
    }

    let sides = two_sides(&[5, 0, 5], &[5, 5, 0]);
    let profile = Profile {
        scorers: vec![WeightedScorer {
            scorer: Box::new(Unwinds),
            weight: 1.0,
        }],
        ..least_requests_profile()
    };
    let engine = pair_session_engine(vec![profile], Assignment::Delegate);
    let keyed = pair_request(Some(7));
    let sessions = side_sessions(&keyed);
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let decision = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    let pick = decision.pick().expect("a pair");
    let target = *pick.targets.first().expect("a target");
    let pins = || sessions.map(|side| engine.sessions.pins(side));
    let none = Some([WorkerId(0); 2]);
    assert!(pins().iter().all(|side| *side != none), "each side claims");
    // Both sides are issued: the decode's lease is the target's.
    let prefill = Lease {
        side: Side::Prefill,
        session: sessions[0],
        session_verdict: pick.session_verdict,
        ..Lease::default()
    };
    let held = HeldDecision {
        decision: &decision,
        lease: &prefill,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: sessions[1],
            tokens: 0,
            chars: 0,
            bytes: 0,
            chosen: false,
        },
    };
    let refused = Said::Released(None, ReleaseReason::CommitFailed);
    let told = catch_unwind(AssertUnwindSafe(|| {
        let hearers = lease_events::of_host(&engine, &engine);
        lease_events::pair(&hearers, &held, refused, |_| 0);
    }));
    assert!(told.is_err(), "the stage unwound");
    assert_eq!(pins(), [none; 2]);
}

/// A pair whose request names its decode is made behind one prefill
/// and no other. Where the decode its number reaches cannot be named,
/// the pair is shed under `target-worker` and leaves no claim: the
/// claim the prefill side's run made is withdrawn, as for any pair
/// that formed none, and the key holds no pin.
#[test]
fn a_pair_whose_named_decode_is_shed_withdraws_the_claim_too() {
    let sides = two_sides(&[0, 5], &[0, 0]);
    let profiles = vec![least_requests_profile(), least_requests_profile()];
    let engine = pair_session_engine(profiles, Assignment::Delegate);
    let keyed = pair_request(Some(7));
    let [prefill_session, _] = side_sessions(&keyed);
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    // Two decodes stand at the places, and the number asks a third.
    let places = [0, 1];
    let decode_cx = Constraints {
        places: Some(&places[..]),
        target_worker: Some(2),
        ..Constraints::default()
    };
    let mut scratch = Scratch::new();
    let missed = engine.decide_pair(&keyed, &pool, pools, &cx, &decode_cx, &mut scratch);
    assert!(
        matches!(&missed, Decision::Shed(shed) if shed.emptied_by == TARGET_WORKER),
        "{missed:?}"
    );
    let none = Some([WorkerId(0); 2]);
    assert_eq!(engine.sessions.pins(prefill_session), none);
}

/// [`pair_session_engine`] under the rule's cap of two, where the
/// profile places a new session.
fn capped_pair_engine(profiles: Vec<Profile>) -> Engine {
    let binding = Binding {
        decode_profile: Some(profiles.len() as u16 - 1),
        sessions: Some(SESSIONS),
        ..pair_binding()
    };
    Engine::new(profiles, vec![binding], 1, Needs::default())
}

/// Three prefills and two decodes for a key at its cap on the prefill
/// side. The key was served on the first prefill, the busiest, which
/// carries the rule's cap of the key's requests. The idle prefill is of
/// pair class 7 and the other two of class 8, as the two decodes are
/// in turn: a respill places the idle prefill first, and a decode
/// profile that declines class 7 ([`declining_decode_profile`]) pairs
/// the third.
fn capped_prefill_sides(engine: &Engine, keyed: &RequestFacts<'_>) -> [Vec<Candidate>; 2] {
    let mut sides = two_sides(&[9, 0, 3], &[0, 0]);
    let [prefills, decodes] = &mut sides;
    for (row, class) in prefills.iter_mut().zip([8, 7, 8]) {
        row.pair_class = class;
    }
    for (row, class) in decodes.iter_mut().zip([7, 8]) {
        row.pair_class = class;
    }
    let [prefill_session, _] = side_sessions(keyed);
    engine
        .sessions
        .lookup(prefill_session, keyed.now_micros, &SESSIONS);
    let served = prefills[0].id;
    engine
        .sessions
        .dispatched(prefill_session, SessionVerdict::Pinned, served);
    prefills[0].session_in_flight = SESSIONS.inflight_cap;
    sides
}

/// A prefill a capped pin's respill placed first and the decode side
/// declines is passed over under a cap too, and the pin's one move
/// goes with the pair. The side's run moves no pin. The pair forms on
/// the next prefill, and that one leads the key's pins where the pair
/// forms, ahead of the pin the key left and ahead of any dispatch. The
/// prefill passed over is no pin. So the key's second request, decided
/// before the first is dispatched, follows a prefill that pairs. The
/// first one's dispatch there moves nothing more. A pair that forms on
/// a pin taken as a pin moves none: the key's failover, taken while
/// the prefill that leads is not ready, stays the failover.
#[test]
fn a_respill_on_a_prefill_that_is_passed_over_moves_to_the_one_that_pairs() {
    let profiles = vec![least_requests_profile(), declining_decode_profile()];
    let engine = capped_pair_engine(profiles);
    let keyed = pair_request(Some(7));
    let [prefill_session, _] = side_sessions(&keyed);
    let mut sides = capped_prefill_sides(&engine, &keyed);
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let (left, paired) = (sides[0][0].id, sides[0][2].id);
    let first = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    let first = first.pick().expect("a pair");
    let went = *first.targets.first().expect("a target");
    assert_eq!(went.id, paired);
    assert_eq!(first.session_verdict, SessionVerdict::Respill);
    let moved = Some([paired, left]);
    assert_eq!(engine.sessions.pins(prefill_session), moved);
    let second = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    let second = second.pick().expect("the key's second pair");
    assert_eq!(second.targets.first().map(|target| target.id), Some(paired));
    assert_eq!(second.session_verdict, SessionVerdict::Pinned);
    // The first decision's dispatch on the prefill that paired.
    let lease = Lease {
        session: prefill_session,
        session_verdict: first.session_verdict,
        ..Lease::default()
    };
    let went = Target { pair: None, ..went };
    engine.on_dispatched(&lease, &went, Side::Prefill, 0);
    assert_eq!(engine.sessions.pins(prefill_session), moved);
    // The prefill that leads is not ready, and the pin the key left
    // is under its cap again.
    sides[0][2].health.healthy = false;
    sides[0][0].session_in_flight = 0;
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let third = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    let third = third.pick().expect("the key's third pair");
    assert_eq!(third.targets.first().map(|target| target.id), Some(left));
    assert_eq!(third.session_verdict, SessionVerdict::Pinned);
    assert_eq!(engine.sessions.pins(prefill_session), moved);
}

/// A pair's decision that forms no pair moves no pin. Its prefill
/// side's respill placed a prefill, which took no request: the key's
/// pins stand as the decision read them, and the key's next request
/// meets the pin at its cap again.
#[test]
fn a_pair_that_sheds_moves_no_pin_for_its_prefill_sides_respill() {
    let profiles = vec![least_requests_profile(), declining_decode_profile()];
    let engine = capped_pair_engine(profiles);
    let keyed = pair_request(Some(7));
    let [prefill_session, _] = side_sessions(&keyed);
    let mut sides = capped_prefill_sides(&engine, &keyed);
    // Every decode is of the refused class: no prefill pairs.
    for row in sides.iter_mut().flatten() {
        row.pair_class = 7;
    }
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let declined = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    assert!(declined.pick().is_none(), "{declined:?}");
    let stands = Some([sides[0][0].id, WorkerId(0)]);
    assert_eq!(engine.sessions.pins(prefill_session), stands);
}

/// What [`DecidesBeside`] stages, and what its request read.
#[derive(Debug, Default)]
struct Beside {
    /// The table, the session and the worker of the staged request.
    staged: Option<(Arc<SessionTable>, Session, WorkerId)>,
    /// The pins that request read of its session.
    read: Option<[WorkerId; 2]>,
}

/// [`RefusesClass`], with a second request of one session decided
/// inside a pair's decision. Where the picker declines, between the
/// prefill side's run and the pair's forming, the staged request reads
/// its session's pins, moves the capped pin to its worker and is sent
/// there. It is staged once.
#[derive(Debug)]
struct DecidesBeside(RefusesClass, Arc<Mutex<Beside>>);

impl Lifecycle for DecidesBeside {}

impl Picker for DecidesBeside {
    fn name(&self) -> &'static str {
        "decides-beside"
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
        self.0.pick(facts, candidates, kept, scores, band, out);
        if !out.is_empty() {
            return;
        }
        let mut beside = self.1.lock().expect("the staged request");
        if let Some((table, session, worker)) = beside.staged.take() {
            beside.read = Some(table.lookup(session, facts.now_micros, &SESSIONS));
            table.respilled(session, worker);
            table.dispatched(session, SessionVerdict::Respill, worker);
        }
    }
}

/// The hole a move written ahead of the pair's forming leaves. A
/// second request of the session is decided and sent inside the
/// pair's decision, between the prefill side's run and the forming.
/// It reads the pins as they stood, with no pin on the prefill the
/// side placed first, and takes that prefill by a move of its own.
/// The pair forms on the next prefill and fronts it as a respill's
/// move does. Nothing is withdrawn behind the second request: the
/// prefill it was sent to stands behind the one that paired.
#[test]
fn a_request_decided_inside_a_pairs_decision_reads_no_move_and_keeps_its_pin() {
    let beside = Arc::new(Mutex::new(Beside::default()));
    let decode = Profile {
        picker: Box::new(DecidesBeside(RefusesClass(7), Arc::clone(&beside))),
        ..declining_decode_profile()
    };
    let engine = capped_pair_engine(vec![least_requests_profile(), decode]);
    let keyed = pair_request(Some(7));
    let [prefill_session, _] = side_sessions(&keyed);
    let sides = capped_prefill_sides(&engine, &keyed);
    let [left, passed_over, paired] = [0, 1, 2].map(|at: usize| sides[0][at].id);
    let staged = (Arc::clone(&engine.sessions), prefill_session, passed_over);
    beside.lock().expect("the staged request").staged = Some(staged);
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let decision = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    let pick = decision.pick().expect("a pair");
    assert_eq!(pick.targets.first().map(|target| target.id), Some(paired));
    assert_eq!(pick.session_verdict, SessionVerdict::Respill);
    let read = beside.lock().expect("the staged request").read;
    assert_eq!(
        read,
        Some([left, WorkerId(0)]),
        "no pin ahead of the forming"
    );
    let behind = Some([paired, passed_over]);
    assert_eq!(engine.sessions.pins(prefill_session), behind);
}

/// A decode side under a capped rule hands its key on at the cap among
/// the partners of the prefill that pairs. The decode taken leads the
/// decode side's pins where the side is decided, ahead of any
/// dispatch, the pair states the respill as that side's verdict, and
/// the decode at its cap is no alternate.
#[test]
fn a_decode_side_at_its_cap_respills_among_its_partners_where_it_is_decided() {
    let profiles = vec![least_requests_profile(), least_requests_profile()];
    let engine = capped_pair_engine(profiles);
    let keyed = pair_request(Some(7));
    let [_, decode_session] = side_sessions(&keyed);
    // The key's decode side was served on the first decode, the
    // busier one, which carries the cap.
    let mut sides = two_sides(&[0, 5], &[9, 5]);
    let (served, taken) = (sides[1][0].id, sides[1][1].id);
    engine
        .sessions
        .lookup(decode_session, keyed.now_micros, &SESSIONS);
    engine
        .sessions
        .dispatched(decode_session, SessionVerdict::Pinned, served);
    sides[1][0].session_in_flight = SESSIONS.inflight_cap;
    let pools = PairPools {
        prefills: &sides[0],
        decodes: &sides[1],
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let decision = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
    let pick = decision.pick().expect("a pair");
    assert_eq!(
        pick.targets.len(),
        1,
        "the decode at its cap is not offered"
    );
    let pair = pick.targets.first().and_then(|target| target.pair);
    let pair = pair.expect("a decode");
    assert_eq!(pair.decode_id, taken);
    assert_eq!(pair.decode_verdict, DecodeVerdict::RESPILL);
    let moved = Some([taken, served]);
    assert_eq!(engine.sessions.pins(decode_session), moved);
}

/// A decode side its rule places writes the decode side's buffers, as
/// its profile's run does, and the prefill's buffer stays as the
/// prefill's run left it, under a rule as under none: a lifted veto's
/// mark on the prefill's buffer is the prefill's own. So the prefill's
/// shadow speaks beside a pair whose decode side alone was steered
/// past the overload veto. Beside a prefill its rule placed it stands
/// aside, whatever the buffer says.
#[test]
fn a_decode_side_under_a_rule_leaves_the_prefills_buffer_to_the_prefill() {
    let engine = |sessions: Option<SessionRule>| {
        let binding = Binding {
            shadow: Some(1),
            sessions,
            ..pair_binding()
        };
        let profiles = vec![strict_profile(), strict_profile()];
        Engine::new(profiles, vec![binding], 1, Needs::default())
    };
    // Every decode is over its thresholds: the binding's rule steers
    // the decode side, and the pair's trace names the lifted veto.
    let [prefills, mut decodes] = two_sides(&[0, 0], &[0, 0]);
    for decode in &mut decodes {
        decode.health.overloaded = true;
    }
    let rule = SessionRule {
        inflight_cap: 0,
        ..SESSIONS
    };
    for (sessions, key) in [(None, None), (Some(rule), Some(7))] {
        let (engine, facts) = (engine(sessions), pair_request(key));
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
        let pick = live.pick().expect("a pair");
        assert_eq!(pick.trace.lifted, LiftedVeto::Overload, "{key:?}");
        assert_eq!(scratch.single.lifted, LiftedVeto::None, "{key:?}");
        let side = PairSide::Prefill;
        let shadow =
            engine.shadow_side(0, side, &facts, &pool, pools, &cx, &cx, pick, &mut scratch);
        let speaks = shadow.is_some_and(|shadow| shadow.pick().is_some());
        assert_eq!(speaks, key.is_none(), "{key:?}");
    }
}

/// Beside a side its binding's sessions rule placed, the engine runs
/// no shadow of that side, at the pair's decision and at a decode
/// decided again: a shadow's run is put to no rule, and its bare
/// profile beside a key's pin would say nothing true of the profile. A
/// request with no key under the same rule is its profiles', and each
/// run speaks. A side whose picker leaves the placement to the
/// assignment is placed so with a key and with none: its shadow is run
/// for no request, and the other side's is, by the live side's own
/// profile.
#[test]
fn a_side_its_sessions_rule_placed_is_run_no_shadow() {
    let [prefills, decodes] = two_sides(&[0, 5], &[0, 5]);
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    // Which runs speak: each side's at the pair's decision, then the
    // later decode's.
    let ran = |engine: &Engine, key: Option<u64>| {
        let facts = pair_request(key);
        let mut scratch = Scratch::new();
        let live = engine.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
        let pick = live.pick().expect("a pair");
        let held = pick.targets.first().expect("the pair's first target").id;
        let sides = PairSide::BOTH.map(|side| {
            let shadow =
                engine.shadow_side(0, side, &facts, &pool, pools, &cx, &cx, pick, &mut scratch);
            shadow.is_some()
        });
        let later =
            engine.shadow_decode_later(0, &facts, &pool, pools, held, None, &cx, &mut scratch);
        (sides, later.is_some())
    };
    let rule = SessionRule {
        inflight_cap: 0,
        ..SESSIONS
    };
    let ruled = Binding {
        shadow: Some(1),
        decode_shadow: Some(2),
        sessions: Some(rule),
        ..pair_binding()
    };
    let profiles = (0..3).map(|_| least_requests_profile()).collect();
    let engine = Engine::new(profiles, vec![ruled], 1, Needs::default());
    assert_eq!(ran(&engine, Some(7)), ([false; 2], false));
    assert_eq!(ran(&engine, None), ([true; 2], true));
    let assigned = Binding {
        decode_profile: Some(3),
        shadow: Some(1),
        decode_shadow: Some(2),
        ..pair_binding()
    };
    let mut profiles = vec![manual_profile()];
    profiles.extend((0..3).map(|_| least_requests_profile()));
    let engine = Engine::new(profiles, vec![assigned], 1, Needs::default());
    assert!(engine.shadows_side(0, PairSide::Prefill), "named and free");
    assert_eq!(ran(&engine, None), ([false, true], true));
}

/// A later decode of a key follows the key's decode pin: the call
/// names the decode the pair's first decision claimed, whether or not
/// it is handed that decode, and states the pin in the first target's
/// pair. The pick states no verdict: the call decided no prefill.
#[test]
fn a_later_decode_follows_its_keys_pin_and_states_it_in_its_pair() {
    let engine = pair_session_engine(vec![least_requests_profile()], Assignment::Delegate);
    let mut scratch = Scratch::new();
    let first = two_sides(&[0], &[5, 0, 5]);
    let keyed = pair_request(Some(7));
    let (places, _) = pair_placed(&engine, &keyed, &first, &mut scratch);
    assert_eq!(places, (0, 1));
    // The loads point at the third decode by the later call.
    let [prefills, decodes] = two_sides(&[0], &[5, 9, 0]);
    let named = (decodes[1].id, decodes[1].dp_rank);
    for held in [Some(named), None] {
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let again = engine.decide_decode(
            &keyed,
            &pool,
            pools,
            prefills[0].id,
            held,
            &cx,
            &mut scratch,
        );
        let again = again.pick().expect("a decode decided again");
        let pair = again.targets.first().and_then(|target| target.pair);
        let pair = pair.expect("a decode");
        assert_eq!(
            (pair.decode_position, pair.decode_verdict),
            (1, DecodeVerdict::PINNED)
        );
        assert_eq!(again.session_verdict, SessionVerdict::None);
    }
    // With no key the later call follows the loads and states nothing.
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let plain = pair_request(None);
    let again = engine.decide_decode(
        &plain,
        &pool,
        pools,
        prefills[0].id,
        None,
        &cx,
        &mut scratch,
    );
    let pair = again.pick().and_then(|pick| pick.targets.first()?.pair);
    let pair = pair.expect("a decode");
    assert_eq!(
        (pair.decode_position, pair.decode_verdict),
        (2, DecodeVerdict::NONE)
    );
}

/// Under a sessions rule each side of a pair states who placed it, by
/// its own decision, and under the rule's word `untold` that it is
/// kept from the profile. A new key's pair is the stages' on both
/// sides under `delegate`. A key pinned on both sides follows both
/// pins, and its pair is the rule's on both. With the worker of its
/// prefill pin down the stages place the prefill beside a decode the
/// rule placed, and the other way round with the worker of its decode
/// pin down. A decode asked again states its own side's word in its
/// pairs and the stages for the prefill: the call decided none. A
/// request with no key is the stages' on both sides. Under the rule's
/// default word the same sides are the rule's, and none is kept from
/// the profile.
///
/// A gateway's sticky map tags a key by its role
/// (`model_gateway/src/policies/registry.rs:762-769`) and selects a
/// pair's two roles apart
/// (`model_gateway/src/routers/common/placement.rs:3026`, `:3088`): a
/// role whose pin stands under its cap asks its policy nothing
/// (`registry.rs:794-797`), beside a role its policy places.
#[test]
fn each_side_of_a_pair_states_who_placed_it_under_the_rules_word() {
    use decision::PlacedBy;

    // Each side's word of a pick: the prefill side's, then the decode
    // side's, which every pair of the pick states.
    let words = |decision: &Decision| {
        let pick = decision.pick().expect("a pair");
        let mut pairs = pick.targets.iter().filter_map(|target| target.pair);
        let decode = pairs.next().expect("a decode").decode_placing;
        assert!(pairs.all(|pair| pair.decode_placing == decode));
        assert_eq!(pick.decode_untold(), decode.untold());
        [
            (pick.placed_by(), pick.untold()),
            (decode.placed_by(), decode.untold()),
        ]
    };
    for kept in [RulePlaced::Untold, RulePlaced::Told] {
        let rule = SessionRule {
            rule_placed: kept,
            inflight_cap: 0,
            ..SESSIONS
        };
        let binding = Binding {
            sessions: Some(rule),
            ..pair_binding()
        };
        let profiles = vec![least_requests_profile()];
        let engine = Engine::new(profiles, vec![binding], 1, Needs::default());
        let mut scratch = Scratch::new();
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let stages = (PlacedBy::Stages, false);
        let ruled = (PlacedBy::Rule, kept == RulePlaced::Untold);
        // The least loaded prefill is the second, the least loaded
        // decode the third.
        let mut rows = two_sides(&[5, 0, 5], &[5, 5, 0]);
        let new = pair_request(Some(6));
        let [prefills, decodes] = &rows;
        let pools = PairPools {
            prefills,
            decodes,
            ..PairPools::default()
        };
        let first = engine.decide_pair(&new, &pool, pools, &cx, &cx, &mut scratch);
        assert_eq!(words(&first), [stages; 2], "{kept:?}: a new key");
        // What each side states for a key pinned on both sides, with
        // the worker of one side's pin down where `down` names one:
        // of the key's next pair, and of its decode asked again
        // behind the pinned prefill. Each form under a key of its own:
        // a side the stages place is claimed for its key.
        let downs = [None, Some(0), Some(1)];
        let forms = downs
            .into_iter()
            .flat_map(|down| [(down, false), (down, true)]);
        for (nth, (down, again)) in forms.enumerate() {
            let case = format!("{kept:?}, side {down:?} down, again {again}");
            let keyed = pair_request(Some(7 + nth as u64));
            let (pinned, _) = pair_placed(&engine, &keyed, &rows, &mut scratch);
            assert_eq!(pinned, (1, 2), "{case}");
            if let Some(side) = down {
                rows[side][[1, 2][side]].health.healthy = false;
            }
            let [prefills, decodes] = &rows;
            let pools = PairPools {
                prefills,
                decodes,
                ..PairPools::default()
            };
            let by = |side: usize| if down == Some(side) { stages } else { ruled };
            let (held, scratch) = (prefills[1].id, &mut scratch);
            let (decided, prefill) = if again {
                // The call decided no prefill: the stages' word.
                let decided = engine.decide_decode(&keyed, &pool, pools, held, None, &cx, scratch);
                (decided, stages)
            } else {
                let decided = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, scratch);
                (decided, by(0))
            };
            assert_eq!(words(&decided), [prefill, by(1)], "{case}");
            if let Some(side) = down {
                rows[side][[1, 2][side]].health.healthy = true;
            }
        }
        let [prefills, decodes] = &rows;
        let pools = PairPools {
            prefills,
            decodes,
            ..PairPools::default()
        };
        let plain = pair_request(None);
        let unkeyed = engine.decide_pair(&plain, &pool, pools, &cx, &cx, &mut scratch);
        assert_eq!(words(&unkeyed), [stages; 2], "{kept:?}: no key");
    }
}
