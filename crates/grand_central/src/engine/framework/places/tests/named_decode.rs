//! The cases of a decode a pair's request names by a number
//! ([`Constraints::target_worker`]): the number is counted among the decode
//! side's places, behind the one prefill a decision picks, in every pass
//! of that side and in no other set.

use super::*;
use crate::{
    document::spec::OverloadSpec,
    engine::{
        decision::{ShedReason, TARGET_WORKER},
        framework::PairPools,
    },
};

/// The stock pair of a round robin prefill beside a hash ring decode.
/// `shed` writes the binding's word for a pool that is all vetoed.
fn pair_engine(shed: bool) -> Engine {
    let mut spec = PolicySpec::pair_presets("round_robin", "consistent_hashing").expect("presets");
    if shed {
        spec.bindings[0].overload = Some(OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        });
    }
    spec.validate().expect("the presets build")
}

/// A ready row at `position` in the pair class `class`: classes pair by
/// their number under no table.
fn classed(position: u32, class: u32) -> Candidate {
    Candidate {
        pair_class: class,
        ..ready(position)
    }
}

/// A decode side held to `places` whose request names the decode at
/// `number` among them.
fn naming(places: &[u32], number: u32) -> Constraints<'_> {
    Constraints {
        target_worker: Some(number),
        ..stating(places)
    }
}

/// The pair `engine` decides over the two pools, the decode side under
/// `decode_cx`.
fn decide_pair(
    engine: &Engine,
    prefills: &[Candidate],
    decodes: &[Candidate],
    decode_cx: &Constraints<'_>,
) -> Decision {
    let pools = PairPools {
        prefills,
        decodes,
        ..PairPools::default()
    };
    let (request, pool) = (facts(1), of(decodes));
    let prefill_cx = Constraints::default();
    engine.decide_pair(
        &request,
        &pool,
        pools,
        &prefill_cx,
        decode_cx,
        &mut Scratch::new(),
    )
}

/// The pair a pick names, its one target: the prefill's place, then the
/// decode's.
fn pair_of(decision: &Decision) -> (u32, u32) {
    let pick = decision.pick().expect("a pick");
    assert_eq!(pick.targets.len(), 1, "a named decode is the one target");
    let target = pick.targets.first().expect("a target");
    let decode = target.pair.expect("a pair's target").decode_position;
    (target.position, decode)
}

/// The reason and the stage of a shed.
fn shed_of(decision: &Decision) -> (ShedReason, &'static str) {
    match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("not a shed: {other:?}"),
    }
}

/// The number counts, in the decode list's order, the decode side's places
/// that the chosen prefill's class pairs with, and the pair's decode is
/// the one it reaches. A place of another class is not counted, and a row
/// outside the places is not either. A number past them sheds the pair
/// with no endpoint under `target-worker`.
#[test]
fn a_number_is_counted_among_the_places_the_picks_class_keeps() {
    let engine = pair_engine(false);
    let prefills = [classed(0, 1)];
    // Decodes 0, 2 and 3 pair with the prefill, and decode 1 does not.
    let decodes = [classed(0, 1), classed(1, 2), classed(2, 1), classed(3, 1)];
    let at = |places: &[u32], number: u32| {
        decide_pair(&engine, &prefills, &decodes, &naming(places, number))
    };
    let every = [0, 1, 2, 3];
    assert_eq!(pair_of(&at(&every, 0)), (0, 0));
    assert_eq!(pair_of(&at(&every, 1)), (0, 2), "past the other class");
    assert_eq!(pair_of(&at(&every, 2)), (0, 3));
    let missed = (ShedReason::NoEndpoints, TARGET_WORKER);
    assert_eq!(shed_of(&at(&every, 3)), missed, "three are counted");
    assert_eq!(pair_of(&at(&[2, 3], 0)), (0, 2), "among the places alone");
    assert_eq!(shed_of(&at(&[2, 3], 2)), missed);
    assert_eq!(shed_of(&at(&every, u32::MAX)), missed);
}

/// A number that reaches no partner of the prefill picked sheds the pair,
/// though the next prefill of the pick has a partner at that number: the
/// partners of no other prefill are asked. Prefill 0 pairs with one decode
/// and prefill 1 with two, and the prefill's profile takes its turns.
#[test]
fn a_number_is_counted_behind_one_prefill_and_no_other() {
    let engine = pair_engine(false);
    let prefills = [classed(0, 1), classed(1, 2)];
    let decodes = [classed(0, 1), classed(1, 2), classed(2, 2)];
    let (mut placed, mut missed) = (0, 0);
    for _ in 0..6 {
        let decision = decide_pair(&engine, &prefills, &decodes, &naming(&[0, 1, 2], 1));
        if decision.pick().is_some() {
            assert_eq!(
                pair_of(&decision),
                (1, 2),
                "the second partner of prefill 1"
            );
            placed += 1;
        } else {
            assert_eq!(shed_of(&decision), (ShedReason::NoEndpoints, TARGET_WORKER));
            missed += 1;
        }
    }
    assert!(placed > 0 && missed > 0, "{placed} placed, {missed} shed");
}

/// The decode a number reaches is the decode side's one row in every pass.
/// One with no free room is taken, and one the request tried is handed
/// again: a soft filter steps aside where it would leave nobody. One over
/// its thresholds is put to the binding's rule for a pool that is all
/// vetoed, which sees that one row: where the rule steers it is taken with
/// the veto lifted, and where the rule sheds the pair is shed under
/// `target-worker`. The other decode, which stands, is never the answer.
/// The decode decided again answers a decode that is not ready so too.
#[test]
fn the_decode_a_number_reaches_is_the_sides_one_row_in_every_pass() {
    let prefills = [classed(0, 1)];
    let first = naming(&[0, 1], 0);
    let steering = pair_engine(false);

    let mut decodes = [classed(0, 1), classed(1, 1)];
    decodes[0].has_room = false;
    let full = decide_pair(&steering, &prefills, &decodes, &first);
    assert_eq!(pair_of(&full), (0, 0), "the decode with no free room");

    let decodes = [classed(0, 1), classed(1, 1)];
    let tried = [decodes[0].id];
    let retry = RequestFacts {
        tried: &tried,
        attempt: 1,
        ..facts(1)
    };
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let again = steering.decide_pair(
        &retry,
        &of(&decodes),
        pools,
        &Constraints::default(),
        &first,
        &mut Scratch::new(),
    );
    assert_eq!(pair_of(&again), (0, 0), "the decode the request tried");

    let mut decodes = [classed(0, 1), classed(1, 1)];
    decodes[0].health.overloaded = true;
    let lifted = decide_pair(&steering, &prefills, &decodes, &first);
    assert_eq!(pair_of(&lifted), (0, 0), "the rule sees the one decode");
    assert_eq!(lifted.trace().lifted, LiftedVeto::Overload);
    let shed = decide_pair(&pair_engine(true), &prefills, &decodes, &first);
    assert_eq!(shed_of(&shed), (ShedReason::NoEndpoints, TARGET_WORKER));

    decodes[0].health = Health::default();
    let gone = decide_pair(&steering, &prefills, &decodes, &first);
    assert_eq!(shed_of(&gone), (ShedReason::NoEndpoints, TARGET_WORKER));
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let called = steering.decide_decode(
        &facts(1),
        &of(&decodes),
        pools,
        prefills[0].id,
        None,
        &first,
        &mut Scratch::new(),
    );
    assert_eq!(shed_of(&called), (ShedReason::NoEndpoints, TARGET_WORKER));
}

/// A number stands beside the decode side's places: a pair's decision that
/// states one beside none is declined under `target-worker`, under a
/// decode side that places by a routing value too, and so is the decode
/// decided again. A decision that states none beside no places answers as
/// it did.
#[test]
fn a_number_beside_no_places_is_declined_at_the_pair_entries() {
    let engine = pair_engine(false);
    let (prefills, decodes) = ([classed(0, 1)], [classed(0, 1), classed(1, 1)]);
    let beside_none = Constraints {
        target_worker: Some(0),
        ..Constraints::default()
    };
    let keyed = RequestFacts {
        session_key: Identity::plain("talk-1"),
        ..facts(1)
    };
    let pools = || PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    let declined = (ShedReason::PolicyDeclined, TARGET_WORKER);
    let prefill_cx = Constraints::default();
    for request in [facts(1), keyed] {
        let pair = |decode_cx: &Constraints<'_>| {
            let mut scratch = Scratch::new();
            engine.decide_pair(
                &request,
                &of(&decodes),
                pools(),
                &prefill_cx,
                decode_cx,
                &mut scratch,
            )
        };
        assert_eq!(shed_of(&pair(&beside_none)), declined);
        assert!(pair(&Constraints::default()).pick().is_some());
        let decode = |decode_cx: &Constraints<'_>| {
            let mut scratch = Scratch::new();
            engine.decide_decode(
                &request,
                &of(&decodes),
                pools(),
                prefills[0].id,
                None,
                decode_cx,
                &mut scratch,
            )
        };
        assert_eq!(shed_of(&decode(&beside_none)), declined);
        assert_eq!(pair_of(&decode(&naming(&[0, 1], 1))), (0, 1));
        let missed = (ShedReason::NoEndpoints, TARGET_WORKER);
        assert_eq!(shed_of(&decode(&naming(&[0, 1], 2))), missed);
    }
}

/// No other set reads the number: under every preset a single request's
/// decision that states one answers as the same decision that states none,
/// held to places or not.
#[test]
fn a_single_requests_decision_reads_no_number() {
    let rows = pool(8);
    let places = [1, 4, 6];
    for name in PRESETS {
        let decide = |cx: &Constraints<'_>| {
            // An engine a decision: a rotation is turned by each.
            let engine = stock(name, false);
            let decision = engine.decide(&facts(7), &of(&rows), &rows, cx, &mut Scratch::new());
            named(&decision)
        };
        let numbered = |cx: Constraints<'static>| Constraints {
            target_worker: Some(1),
            ..cx
        };
        let whole = Constraints::default();
        assert_eq!(decide(&numbered(whole)), decide(&whole), "{name}");
        let held = Constraints {
            places: Some(&[1, 4, 6]),
            ..Constraints::default()
        };
        let among = decide(&held);
        assert!(among.iter().all(|at| places.contains(at)), "{name}");
        assert_eq!(decide(&numbered(held)), among, "{name}");
    }
}
