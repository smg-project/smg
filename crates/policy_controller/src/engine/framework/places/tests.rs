//! The cases of a request held to places of its pool, for the passes that
//! hold it: the staged pass hands no row outside the places, under a
//! profile that lists no filter and under the crate's presets as they ship.

use super::super::{Binding, Engine, Plan, Profile};
use crate::{
    document::spec::{Needs, PolicySpec, PRESETS},
    engine::{
        decision::{Decision, LiftedVeto},
        view::{Candidate, Constraints, Health, PoolFacts, RequestFacts, WorkerId},
        Scratch,
    },
    policies::random::Random,
};

/// A ready row at `position`, under an id and a stable key of its own.
fn ready(position: u32) -> Candidate {
    Candidate {
        position,
        id: WorkerId(100 + u64::from(position)),
        key: 100 + position,
        health: Health {
            healthy: true,
            ..Health::default()
        },
        has_capacity: true,
        has_room: true,
        ..Candidate::default()
    }
}

fn pool(rows: u32) -> Vec<Candidate> {
    (0..rows).map(ready).collect()
}

/// The preset `name` as it ships, or with no filter in any of its profiles
/// where `bare`: a valid document that weighs no worker's health.
fn stock(name: &str, bare: bool) -> Engine {
    let mut spec = PolicySpec::preset(name).expect("a preset");
    if bare {
        for profile in spec.profiles.values_mut() {
            profile.filter.clear();
        }
    }
    spec.validate().expect("the document builds")
}

fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        seed,
        session_counted: true,
        ..RequestFacts::default()
    }
}

/// The facts of a pool that lists no engine.
fn of(rows: &[Candidate]) -> PoolFacts<'static> {
    PoolFacts {
        pool_size: rows.len() as u32,
        generation: 1,
        ..PoolFacts::default()
    }
}

/// A request held to `places`.
fn stating(places: &[u32]) -> Constraints<'_> {
    Constraints {
        places: Some(places),
        ..Constraints::default()
    }
}

/// The places a decision's targets name, best first; none for a shed.
fn named(decision: &Decision) -> Vec<u32> {
    let targets = decision.pick().map(|pick| pick.targets.iter());
    targets
        .into_iter()
        .flatten()
        .map(|target| target.position)
        .collect()
}

/// Every preset, with its filters and with none, over a pool under the
/// draw's floor, for a request with no key: no target stands outside the
/// places, and the presets that place a request by its pool alone place
/// every one. A preset with its filters folds them into one test of a row,
/// asked in one pass (`keep_folded`): that pass steps aside for a request
/// held to places, and the staged pass decides.
#[test]
fn no_preset_hands_a_target_outside_the_places_under_the_draws_floor() {
    let places = [2, 5];
    let by_pool = [
        "random",
        "round_robin",
        "passthrough",
        "power_of_two",
        "least_load",
        "consistent_hashing",
    ];
    let rows = pool(8);
    for name in PRESETS {
        for bare in [false, true] {
            let engine = stock(name, bare);
            for seed in 0..24 {
                let (cx, mut scratch) = (stating(&places), Scratch::new());
                let decision = engine.decide(&facts(seed), &of(&rows), &rows, &cx, &mut scratch);
                let named = named(&decision);
                let what = format!("{name}, bare {bare}, seed {seed}");
                assert!(
                    named.iter().all(|place| places.contains(place)),
                    "{what}: {named:?}"
                );
                if by_pool.contains(&name) {
                    assert!(!named.is_empty(), "{what}: a kept worker takes it");
                }
            }
        }
    }
}

/// The filter half: a profile of no filter over a pool under the draw's
/// floor is the whole pass's, and its set opens with the places. Each of
/// them takes requests, and a place past the pool's end names no row.
#[test]
fn the_filter_half_opens_with_the_places_under_a_profile_of_no_filter() {
    let profile = Profile {
        name: "random".into(),
        filters: vec![],
        scorers: vec![],
        picker: Box::new(Random),
    };
    let binding = Binding {
        model: "*".into(),
        phases: vec![],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow: None,
        decode_shadow: None,
        sessions: None,
        overload: None,
    };
    let engine = Engine::new(vec![profile], vec![binding], 1, Needs::default());
    let rows = pool(8);
    let mut first = [0_u32; 8];
    for seed in 0..64 {
        let (cx, mut scratch) = (stating(&[1, 6, 99]), Scratch::new());
        let decision = engine.decide(&facts(seed), &of(&rows), &rows, &cx, &mut scratch);
        // The whole pass counts the survivors of its filters: the two
        // places of the pool, and not the one past its end.
        let survivors = decision.pick().and_then(|pick| pick.trace.survivors.get());
        assert_eq!(survivors, Some(2), "seed {seed}");
        let named = named(&decision);
        assert!(
            named.iter().all(|place| [1, 6].contains(place)),
            "{named:?}"
        );
        first[named[0] as usize] += 1;
    }
    assert!(first[1] > 0 && first[6] > 0, "{first:?}");
    // Held to no place of the pool, the request has no candidate.
    let (cx, mut scratch) = (stating(&[]), Scratch::new());
    let none = engine.decide(&facts(1), &of(&rows), &rows, &cx, &mut scratch);
    assert!(none.pick().is_none());
}

/// A pool whose every worker is over its thresholds is steered among the
/// places: the fewest requests in flight of the workers kept, whatever the
/// workers left out hold.
#[test]
fn a_pool_over_its_thresholds_is_steered_among_the_places() {
    let engine = stock("round_robin", false);
    let mut rows = pool(8);
    for row in &mut rows {
        row.health.overloaded = true;
    }
    for (place, in_flight) in [(1, 9), (4, 2), (6, 5)] {
        rows[place].in_flight = in_flight;
    }
    for seed in 1..=8 {
        let (cx, mut scratch) = (stating(&[1, 4, 6]), Scratch::new());
        let decision = engine.decide(&facts(seed), &of(&rows), &rows, &cx, &mut scratch);
        assert_eq!(named(&decision), [4, 6, 1], "seed {seed}");
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.trace.lifted, LiftedVeto::Overload);
    }
}

/// A request held to the one place it already tried is handed that place
/// again, not refused, as over a list of that worker alone: the filter of
/// tried workers is soft, and steps aside where it would leave nobody.
#[test]
fn a_request_held_to_the_one_place_it_tried_is_handed_it_again() {
    let rows = pool(8);
    let gone = [rows[5].id];
    for name in ["round_robin", "least_load"] {
        let engine = stock(name, false);
        let request = RequestFacts {
            attempt: 1,
            tried: &gone,
            ..facts(3)
        };
        let cx = Constraints {
            excluded: &gone,
            ..stating(&[5])
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(&request, &of(&rows), &rows, &cx, &mut scratch);
        assert_eq!(named(&decision), [5], "{name}");
    }
}
