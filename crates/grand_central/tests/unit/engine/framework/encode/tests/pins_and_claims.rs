use super::*;

/// [`engine`] under a sessions rule with no cap, which leaves a new key to
/// the profile and says `word` of a request it places alone.
fn ruled(word: spec::RulePlaced) -> Engine {
    let mut spec = document();
    let stock = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rule = stock.bindings[0].sessions.clone();
    let sessions = rule.as_mut().expect("the stock rule");
    (sessions.assignment, sessions.rule_placed) = (spec::Assignment::Delegate, word);
    spec.view.headers.push(sessions.sticky_header.clone());
    spec.bindings[0].sessions = rule;
    spec.validate().expect("a rule under the three-sided plan")
}

/// Tells `engine`'s session table that both sides of `pick` went out for
/// `facts`, a side at a time as a host commits a pair: each side's lease
/// states that side's own session and its own verdict, the decode side's
/// as the first target's pair keeps it.
fn dispatched(engine: &Engine, facts: &RequestFacts<'_>, pick: &Pick) {
    let target = pick.targets.first().expect("a target");
    let pair = target.pair.expect("a pair");
    let decode = pair.decode_verdict.on(pair.decode_id);
    let sides = [
        (Side::Prefill, Phase::Prefill, pick.session_verdict),
        (Side::Decode, Phase::Decode, decode),
    ];
    for (side, phase, session_verdict) in sides {
        let lease = Lease {
            session: RequestFacts { phase, ..*facts }.session(),
            session_verdict,
            ..Lease::default()
        };
        engine.dispatched_table(&lease, target, side);
    }
}

/// Each side's word of a pick: who placed it, and whether it is kept from
/// its profile. The prefill side's, then the decode side's, which every
/// pair of the pick states.
fn words(decision: &Decision) -> [(PlacedBy, bool); 2] {
    let Decision::Pick(pick) = decision else {
        panic!("a pick, not {decision:?}");
    };
    let mut pairs = pick.targets.iter().filter_map(|target| target.pair);
    let decode = pairs.next().expect("a decode").decode_placing;
    assert!(pairs.all(|pair| pair.decode_placing == decode));
    [
        (pick.placed_by(), pick.untold()),
        (decode.placed_by(), decode.untold()),
    ]
}

/// The request of `bed` under the session key of hash `key`.
fn keyed(bed: &Bed, key: u64) -> RequestFacts<'static> {
    RequestFacts {
        session_key: Identity::hashed(key),
        ..bed.asked().0
    }
}

/// A pick keeps each side's word of who placed it beside a list this side
/// numbered anew, as beside the host's own: the prefills cut to the classes
/// an encoder serves, the decodes cut to the ones that take images, and
/// both. A new key's pair is the stages' on both sides. The key's next
/// request follows both pins, by the host's rows and past the loads: its
/// pair is the rule's on both sides, and under the rule's word `untold`
/// each side is kept from its profile. A list built anew for the host's
/// rows said the stages of the prefill side, and a profile then heard of a
/// request its rule keeps from it.
///
/// A gateway's sticky map follows a pin under its cap among the workers
/// its three-sided selection holds eligible, and asks the role's policy
/// nothing there (`model_gateway/src/policies/registry.rs:833-836`; the
/// eligible workers at
/// `model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2651-2654`).
/// The lines are main's at `9b72a221d`.
#[test]
fn a_rules_pick_states_who_placed_it_beside_a_pool_that_was_cut() {
    let stages = (PlacedBy::Stages, false);
    for word in [spec::RulePlaced::Told, spec::RulePlaced::Untold] {
        let rule = (PlacedBy::Rule, word == spec::RulePlaced::Untold);
        // Which lists are cut: none, the prefills, the decodes, both.
        for (prefills, decodes) in [(false, false), (true, false), (false, true), (true, true)] {
            let case = format!("{word:?}: prefills cut {prefills}, decodes cut {decodes}");
            let engine = ruled(word);
            // Three rows a side, the second the emptiest. A side that is
            // cut leaves its first row out: the first prefill's class has
            // no encoder, and the first decode takes no image.
            let loaded = |role| {
                let rows = [5, 0, 9].into_iter().zip(0..);
                rows.map(|(load, at)| row(at, role, 1, load)).collect()
            };
            let mut bed = Bed::new();
            (bed.prefills, bed.decodes) = (loaded(Role::Prefill), loaded(Role::Decode));
            bed.prefill_classes = if prefills { vec![2, 1, 1] } else { vec![1; 3] };
            (bed.decodes[0].vision, bed.decode_images) = (!decodes, decodes);
            let (facts, pool) = (keyed(&bed, 7), bed.asked().1);
            let (first, _) = bed.decide_asked(&engine, &facts, &pool);
            assert_eq!(pair_of(&first), (1, 1), "{case}");
            assert_eq!(words(&first), [stages; 2], "{case}: a new key");
            dispatched(&engine, &facts, first.pick().expect("a pair"));
            // The loads point away from each side's second row by then.
            (bed.prefills[1].in_flight, bed.decodes[1].in_flight) = (20, 20);
            let (next, _) = bed.decide_asked(&engine, &facts, &pool);
            assert_eq!(pair_of(&next), (1, 1), "{case}: the pins");
            assert_eq!(words(&next), [rule; 2], "{case}");
        }
    }
}

/// An item's answer that turns a request away behind its pair takes back
/// the claims that pair's decision made for the request's key, each side's
/// under its own session. The pair's decision claims a new key's prefill
/// and its decode where it places them, ahead of any dispatch, and the
/// request then goes to neither. So the key's next request finds no pin:
/// the stages place both its sides again, by the loads as they then stand.
/// A claim left standing led that request to the pair the first never
/// reached, as the rule's own. A key that was served keeps its pins behind
/// such an answer: the decision followed them and claimed nothing.
///
/// A gateway's sticky map differs here, and the design records it under
/// its decision 13. The map writes a role's pin where the role is selected
/// (`model_gateway/src/policies/registry.rs:875`, and `:856` at the cap).
/// The three-sided selection picks its pair first
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2655`)
/// and assigns the encoders behind it (`:2700-2707`): an assignment that
/// fails returns with both roles' pins where the pair's selection wrote
/// them. The pair's own selection leaves its prefill's pin so behind a
/// decode its policy declines
/// (`model_gateway/src/routers/common/placement.rs:3147-3148`,
/// `:3210-3212`). The lines are main's at `9b72a221d`.
#[test]
fn an_item_turned_away_behind_its_pair_takes_back_the_pairs_claims() {
    let mut engine = ruled(spec::RulePlaced::Told);
    for profile in engine.profiles.iter_mut().map(own) {
        profile.filters.push(Box::new(RefusesTheKey));
    }
    // Two rows a side, the first the emptier. A request of two items: the
    // first is placed, and the second's stage turns the request away.
    let both = vec![Identity::hashed(11), Identity::hashed(12)];
    let kept = vec![Identity::hashed(11)];
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 5)];
    bed.prefill_classes = vec![1, 1];
    bed.decodes = vec![row(0, Role::Decode, 1, 0), row(1, Role::Decode, 1, 5)];
    bed.keys.clone_from(&both);
    let (new, pool) = (keyed(&bed, 7), bed.asked().1);
    let (turned, _) = bed.decide_asked(&engine, &new, &pool);
    assert_eq!(shed_of(&turned).1, "refuses-the-key");
    // The key's next request carries an item the engine keeps, and the
    // loads point at each side's second row by then.
    bed.keys.clone_from(&kept);
    (bed.prefills[0].in_flight, bed.decodes[0].in_flight) = (9, 9);
    let (next, _) = bed.decide_asked(&engine, &new, &pool);
    assert_eq!(
        pair_of(&next),
        (1, 1),
        "the key followed a pair it never reached"
    );
    assert_eq!(words(&next), [(PlacedBy::Stages, false); 2]);
    // Another key is served on that pair, and the loads then point at each
    // side's first row. Its request that is turned away moves no pin.
    let served = keyed(&bed, 8);
    let (first, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(pair_of(&first), (1, 1));
    dispatched(&engine, &served, first.pick().expect("a pair"));
    (bed.prefills[1].in_flight, bed.decodes[1].in_flight) = (20, 20);
    bed.keys.clone_from(&both);
    let (turned, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(shed_of(&turned).1, "refuses-the-key");
    bed.keys.clone_from(&kept);
    let (next, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(pair_of(&next), (1, 1), "a served key's pins");
    assert_eq!(words(&next), [(PlacedBy::Rule, false); 2]);
}

/// The engine a gateway's settings mean under the encode mode with the
/// manual policy on the pair's two sides: a key the rule has not seen goes
/// to the row that holds the fewest requests.
fn manual_pairs() -> Engine {
    let settings = RoutingSettings {
        mode: ModeSettings::EncodePrefillDecode {
            encode_policy: None,
            prefill_policy: None,
            decode_policy: None,
        },
        policy: PolicySettings::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 14_400,
            assignment_mode: spec::Assignment::MinLoad,
        },
        ..RoutingSettings::default()
    };
    let spec = settings.policy_document();
    let spec = spec.expect("the settings have a document");
    spec.validate().expect("the form validates")
}

/// Each side's session verdict of a pick: the prefill side's, then the
/// decode side's as the first target's pair keeps it.
fn verdicts(decision: &Decision) -> [SessionVerdict; 2] {
    let pick = decision.pick().expect("a pick");
    let pair = pick.targets.first().and_then(|target| target.pair);
    let pair = pair.expect("a pair");
    [pick.session_verdict, pair.decode_verdict.on(pair.decode_id)]
}

/// The claims are taken back so where the rule's own assignment made them:
/// under the manual policy on the pair's two sides, as a gateway's settings
/// say the encode mode. The rule places a new key's prefill and its decode
/// on the rows that hold the fewest requests, and claims both. An item's
/// answer then turns the request away, and the key's next request is placed
/// again, by the loads as they then stand. A key that was served keeps its
/// pins behind such an answer.
///
/// A gateway's `manual` policy differs here, as its sticky map does. It
/// writes a key's entry for a role where it selects that role's worker
/// (`model_gateway/src/policies/manual.rs:357-365`). The three-sided
/// selection picks its pair first
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2660`)
/// and assigns the encoders behind it (`:2705-2712`): an assignment that
/// fails returns with both entries written. The design records the
/// difference under its decision 13. The lines are main's at `13f5cbe29`.
#[test]
fn an_item_turned_away_takes_back_the_claims_the_rules_assignment_made() {
    let mut engine = manual_pairs();
    for profile in engine.profiles.iter_mut().map(own) {
        profile.filters.push(Box::new(RefusesTheKey));
    }
    // Two rows a side, the first the emptier. A request of two items: the
    // first is placed, and the second's stage turns the request away.
    let both = vec![Identity::hashed(11), Identity::hashed(12)];
    let kept = vec![Identity::hashed(11)];
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 5)];
    bed.prefill_classes = vec![1, 1];
    bed.decodes = vec![row(0, Role::Decode, 1, 0), row(1, Role::Decode, 1, 5)];
    bed.keys.clone_from(&both);
    let (new, pool) = (keyed(&bed, 7), bed.asked().1);
    let (turned, _) = bed.decide_asked(&engine, &new, &pool);
    assert_eq!(shed_of(&turned).1, "refuses-the-key");
    // The key's next request carries an item the engine keeps, and the
    // loads point at each side's second row by then. The rule claims that
    // pair: a claim left standing led the key to the first.
    bed.keys.clone_from(&kept);
    (bed.prefills[0].in_flight, bed.decodes[0].in_flight) = (9, 9);
    let (next, _) = bed.decide_asked(&engine, &new, &pool);
    assert_eq!(
        pair_of(&next),
        (1, 1),
        "the key followed a pair it never reached"
    );
    let claimed = [1_001, 2_001].map(|id| SessionVerdict::Claimed(WorkerId(id)));
    assert_eq!(verdicts(&next), claimed);
    // Another key is served on that pair, and the loads then point at each
    // side's first row. Its request that is turned away moves no pin.
    let served = keyed(&bed, 8);
    let (first, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(pair_of(&first), (1, 1));
    dispatched(&engine, &served, first.pick().expect("a pair"));
    (bed.prefills[1].in_flight, bed.decodes[1].in_flight) = (20, 20);
    bed.keys.clone_from(&both);
    let (turned, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(shed_of(&turned).1, "refuses-the-key");
    bed.keys.clone_from(&kept);
    let (next, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(pair_of(&next), (1, 1), "a served key's pins");
    assert_eq!(verdicts(&next), [SessionVerdict::Pinned; 2]);
}
