//! The cases of a frame that states its request's hard demands
//! (`DecideRequest.demands`) and names no worker. They are a pair's frame,
//! a decode call and a frame that stands alone at a side's phase, each
//! asked as a deciding thread asks it. A frame that names its worker has
//! its cases beside the other named ones.

use std::{collections::BTreeSet, sync::Arc, time::Instant};

use smg_worker::load::state::LoadSnapshot;

use super::{
    alone_at, at_call, at_prefill, demanding, holder, pairs_of, preset, seeing_member,
    served_alone, served_call, served_pair, shed_of, stating_member, stock_pair_plan, together,
    Shed, GATEWAY,
};
use crate::{
    engine::{
        decision::ShedReason,
        view::{Role, WireModes, WorkerId},
        DemandFacts,
    },
    service::proto,
    store::MemoryStore,
    tracking::lease::LeaseId,
};

/// The workers a frame's answers named, or the one shed each of them was.
type Named<T> = Result<BTreeSet<T>, Shed>;

/// The shed of a frame whose demand leaves a list nobody: with no
/// endpoint, under the demand's own stage.
const fn unmet(stage: &'static str) -> (ShedReason, &'static str) {
    (ShedReason::NoEndpoints, stage)
}

/// The workers a run of answers named together, or the one shed each of
/// them was ([`together`]).
fn named<T: Ord>(answers: impl Iterator<Item = Result<Vec<T>, Shed>>) -> Named<T> {
    together(answers).map(|picks| picks.into_iter().flatten().collect())
}

/// Every pair of one of `prefills` with one of `decodes`.
fn every(prefills: &[WorkerId], decodes: &[WorkerId]) -> BTreeSet<(WorkerId, WorkerId)> {
    let pairs = prefills
        .iter()
        .flat_map(|&prefill| decodes.iter().map(move |&decode| (prefill, decode)));
    pairs.collect()
}

/// A pair's frame that states a hard demand is answered workers that meet
/// it, each side by what its role is held to. Two of three workers a side
/// state the runtime 2, and one decode alone states a way to decide a
/// constraint.
///
/// A frame that stays on the runtime 2 is answered those two a side. The
/// way holds the decodes and no prefill: its frame is answered any prefill
/// beside the one decode. A way no decode states sheds the frame with no
/// endpoint, under `decoding`. A frame that states no demand is answered
/// every pair.
#[test]
fn a_pairs_frame_that_states_a_demand_is_answered_workers_that_meet_it_on_each_side() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let stating = |name: &str, role, runtime, decoding| {
        let stated = DemandFacts {
            runtime,
            decoding,
            ..DemandFacts::default()
        };
        stating_member(&store, &objects, name, role, None, stated)
    };
    let (prefill, decode) = (Role::Prefill, Role::Decode);
    let prefills = [
        stating("p0", prefill, 1, 0),
        stating("p1", prefill, 2, 0),
        stating("p2", prefill, 2, 0),
    ];
    let decodes = [
        stating("d0", decode, 1, 0),
        stating("d1", decode, 2, 0),
        stating("d2", decode, 2, 3),
    ];
    let now = Instant::now();
    let answered = |demands: proto::Demands| -> Named<(WorkerId, WorkerId)> {
        named((1..=64).map(|n| {
            let frame = demanding(at_prefill(n, WireModes::NONE, None), demands);
            let answer = served_pair(&store, &objects, frame, GATEWAY, now);
            match answer.pick() {
                Some(_) => Ok(pairs_of(&answer)),
                None => Err(shed_of(&answer)),
            }
        }))
    };
    let none = proto::Demands::default();
    assert_eq!(answered(none), Ok(every(&prefills, &decodes)));
    let wire = proto::Demands { runtime: 2, ..none };
    assert_eq!(answered(wire), Ok(every(&prefills[1..], &decodes[1..])));
    let decided = proto::Demands {
        decoding: 3,
        ..none
    };
    assert_eq!(answered(decided), Ok(every(&prefills, &decodes[2..])));
    let other = proto::Demands {
        decoding: 4,
        ..none
    };
    assert_eq!(answered(other), Err(Some(unmet("decoding"))));
}

/// A pair's frame that demands vision is answered workers that take
/// images, on each side. A fleet holds two prefills and two decodes, and
/// at most one worker a side takes images.
///
/// Where one prefill and one decode take images, a frame that demands
/// vision is answered that prefill beside that decode. Where a side holds
/// none that does, the frame is shed with no endpoint, under `vision`. A
/// frame that demands none is answered every pair of each fleet.
#[test]
fn a_pairs_frame_that_demands_vision_is_answered_workers_that_take_images_on_each_side() {
    let (prefill, decode) = (Role::Prefill, Role::Decode);
    let now = Instant::now();
    let none = proto::Demands::default();
    let vision = proto::Demands {
        vision: true,
        ..none
    };
    // Whether a fleet's second prefill takes images, and its second decode.
    for (ahead, behind) in [(true, true), (true, false), (false, true)] {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        store
            .apply(stock_pair_plan())
            .expect("above the unbound start");
        let seeing = |name: &str, role, takes| seeing_member(&store, &objects, name, role, takes);
        let prefills = [seeing("p0", prefill, false), seeing("p1", prefill, ahead)];
        let decodes = [seeing("d0", decode, false), seeing("d1", decode, behind)];
        // Every frame is answered the same way: a pair, or the one shed. A
        // pair named ahead of a shed is not dropped: frames that part fail
        // the case.
        let answered = |demands: proto::Demands| -> Named<(WorkerId, WorkerId)> {
            named((1..=64).map(|n| {
                let frame = demanding(at_prefill(n, WireModes::NONE, None), demands);
                let answer = served_pair(&store, &objects, frame, GATEWAY, now);
                match answer.pick() {
                    Some(_) => Ok(pairs_of(&answer)),
                    None => Err(shed_of(&answer)),
                }
            }))
        };
        assert_eq!(answered(none), Ok(every(&prefills, &decodes)));
        let meeting = if ahead && behind {
            Ok(every(&prefills[1..], &decodes[1..]))
        } else {
            Err(Some(unmet("vision")))
        };
        assert_eq!(
            answered(vision),
            meeting,
            "prefill {ahead}, decode {behind}"
        );
    }
}

/// A decode call that states a hard demand is answered a decode that meets
/// it, whole. It judges no prefill: the held one ran already, and here it
/// states the runtime 1.
///
/// A call that stays on the runtime 2 is answered the two decodes that
/// state it, behind that prefill. A call that states a way to decide is
/// answered the one decode that states it. A way no decode states sheds
/// the call with no endpoint, under `decoding`. A call that states no
/// demand is answered any decode.
#[test]
fn a_decode_call_that_states_a_demand_is_answered_a_decode_that_meets_it() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(stock_pair_plan())
        .expect("above the unbound start");
    let stating = |name: &str, role, runtime, decoding| {
        let stated = DemandFacts {
            runtime,
            decoding,
            ..DemandFacts::default()
        };
        stating_member(&store, &objects, name, role, None, stated)
    };
    let held = stating("p0", Role::Prefill, 1, 0);
    let decodes = [
        stating("d0", Role::Decode, 1, 0),
        stating("d1", Role::Decode, 2, 0),
        stating("d2", Role::Decode, 2, 3),
    ];
    let now = Instant::now();
    let unread = Arc::new(LoadSnapshot::default());
    let answered = |demands: proto::Demands| -> Named<WorkerId> {
        named((1..=32).map(|n| {
            let frame = at_call(n, LeaseId(1), WireModes::NONE, None);
            let frame = demanding(frame, demands);
            let asking = (GATEWAY, now);
            let behind = (held, None);
            let call = served_call(&store, &objects, frame, asking, behind, (&unread, None));
            if call.pick().is_none() {
                return Err(shed_of(&call));
            }
            let decodes = pairs_of(&call).into_iter().map(|(prefill, decode)| {
                assert_eq!(prefill, held, "the held prefill, whatever it states");
                decode
            });
            Ok(decodes.collect())
        }))
    };
    let none = proto::Demands::default();
    assert_eq!(answered(none), Ok(BTreeSet::from(decodes)));
    let wire = proto::Demands { runtime: 2, ..none };
    assert_eq!(answered(wire), Ok(BTreeSet::from([decodes[1], decodes[2]])));
    let decided = proto::Demands {
        decoding: 3,
        ..none
    };
    assert_eq!(answered(decided), Ok(BTreeSet::from([decodes[2]])));
    let other = proto::Demands {
        decoding: 4,
        ..none
    };
    assert_eq!(answered(other), Err(Some(unmet("decoding"))));
}

/// A frame that stands alone at a side's phase and states a hard demand is
/// held as that side's role is. Under a plan of one profile, two workers a
/// side state a window of 4,096 tokens and the third a window of 16. Two
/// decodes state a way to decide a constraint: a wide one and the narrow
/// one.
///
/// A prefill holds the input alone. A prefill frame whose input takes
/// 1,024 tokens is answered the two wide prefills, whatever completion and
/// way it states beside. One that states a completion alone is answered
/// any prefill.
///
/// A decode holds every part. A decode frame is answered the one decode
/// that is wide and states the way. One whose completion no decode's
/// window holds is shed with no endpoint, under `context-window`.
#[test]
fn a_frame_that_stands_alone_and_states_a_demand_is_held_as_its_phases_role_is() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(preset("random", 1))
        .expect("above the unbound start");
    let stating = |name: &str, role, window, decoding| {
        let stated = DemandFacts {
            window,
            decoding,
            ..DemandFacts::default()
        };
        stating_member(&store, &objects, name, role, None, stated)
    };
    let prefills = [
        stating("p0", Role::Prefill, 4_096, 0),
        stating("p1", Role::Prefill, 4_096, 0),
        stating("p2", Role::Prefill, 16, 0),
    ];
    let decodes = [
        stating("d0", Role::Decode, 4_096, 3),
        stating("d1", Role::Decode, 4_096, 0),
        stating("d2", Role::Decode, 16, 3),
    ];
    let now = Instant::now();
    // What sixty-four answers name first, then every target they name: a
    // demand holds the ones behind the first too.
    let answered = |phase: proto::Phase,
                    demands: proto::Demands|
     -> Result<(BTreeSet<WorkerId>, BTreeSet<WorkerId>), Shed> {
        let picks = together((1..=64).map(|n| {
            let frame = demanding(alone_at(n, phase), demands);
            let answer = served_alone(&store, &objects, frame, (GATEWAY, now), None);
            let Some(pick) = answer.pick() else {
                return Err(shed_of(&answer));
            };
            let targets = pick.targets.iter().map(|target| target.id);
            let targets = targets.collect::<Vec<_>>();
            assert!(!targets.is_empty(), "answer {n}: a pick names no worker");
            Ok(targets)
        }))?;
        let first = picks.iter().filter_map(|targets| targets.first().copied());
        let first = first.collect();
        Ok((first, picks.into_iter().flatten().collect()))
    };
    // The workers a run names first are the ones it names at all.
    let both = |workers: BTreeSet<WorkerId>| Ok((workers.clone(), workers));
    let (ahead, behind) = (proto::Phase::Prefill, proto::Phase::Decode);
    let none = proto::Demands::default();
    assert_eq!(answered(ahead, none), both(BTreeSet::from(prefills)));
    assert_eq!(answered(behind, none), both(BTreeSet::from(decodes)));
    let whole = proto::Demands {
        input_window: 1_024,
        output_window: 2_048,
        decoding: 3,
        ..none
    };
    let wide = BTreeSet::from([prefills[0], prefills[1]]);
    assert_eq!(answered(ahead, whole), both(wide), "the input alone");
    let one = BTreeSet::from([decodes[0]]);
    assert_eq!(answered(behind, whole), both(one), "every part");
    let completion = proto::Demands {
        output_window: 8_192,
        ..none
    };
    let any = BTreeSet::from(prefills);
    assert_eq!(
        answered(ahead, completion),
        both(any),
        "no prefill's to hold"
    );
    assert_eq!(
        answered(behind, completion),
        Err(Some(unmet("context-window")))
    );
}
