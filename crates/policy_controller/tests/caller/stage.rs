//! The program's own stage, decided over records a caller makes by hand.
//!
//! A program that adds a stage tests it where it controls every record: it
//! makes the records, a pair's two pools and the targets it expects with
//! the constructors of the crate's types ([`Candidate::ready`],
//! [`PairPools::of`], [`Target::of`], [`Target::paired`]) and writes over
//! them the fields its stage reads. No case here writes a literal of one of
//! those types.
//!
//! The last case places requests through the program, over the lists its
//! registry publishes: the stage reads the same fields there.

use smg_policy_controller::{
    document::{AppliedSpec, PolicySettings, PolicySpec},
    engine::{
        Candidate, Constraints, Decision, Engine, Health, PairPools, Phase, PoolFacts,
        RequestFacts, Role, Scratch, Target, WorkerId,
    },
};

use crate::{
    engine_load::{decided_by_engine_load, stages},
    program::{settings, Asked},
    served, started, workers, MODEL,
};

/// The engine of `document` with the program's scorer in every profile,
/// under the picker that takes the best score.
fn decided(mut document: PolicySpec) -> Engine {
    decided_by_engine_load(&mut document);
    let stages = stages().expect("a free name");
    let engine = document.validate_in(&stages, &[]);
    engine.expect("the catalog holds the scorer")
}

/// A request for [`MODEL`] decided under `seed`.
fn request(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: MODEL,
        phase: Phase::Prefill,
        seed,
        ..RequestFacts::default()
    }
}

/// A rank of the engine at place 0 of its list, at `position`, holding
/// `in_flight` requests.
fn rank(position: u32, role: Role, dp_rank: u16, in_flight: u32) -> Candidate {
    Candidate {
        engine: 0,
        dp_rank,
        dp_size: 2,
        in_flight,
        ..alone(position, role, in_flight)
    }
}

/// A worker that stands alone at `position`, holding `in_flight` requests:
/// a record made ready is an engine of its own. Its engine is its position,
/// which is right where every record before it is an engine of its own.
fn alone(position: u32, role: Role, in_flight: u32) -> Candidate {
    Candidate {
        in_flight,
        pair_class: 1,
        ..Candidate::ready(position, WorkerId(100 + u64::from(position)), role)
    }
}

/// A worker that stands alone behind the two ranks of engine 0, at
/// `position`: the next engine of its list. An engine states its place
/// among the engines of its list, so this one states `1`.
fn behind(position: u32, role: Role, in_flight: u32) -> Candidate {
    Candidate {
        engine: 1,
        ..alone(position, role, in_flight)
    }
}

/// The first target `decision` picks.
fn first(decision: &Decision) -> Target {
    let pick = decision.pick().expect("a pick");
    *pick.targets.first().expect("a target")
}

/// A request goes to the engine that holds the fewest requests over all its
/// ranks: past the idle rank of an engine whose other rank holds three, to
/// a worker that stands alone and holds two. Once the worker alone holds
/// more than the engine does, the request goes to a rank of that engine.
#[test]
fn a_request_goes_past_an_idle_rank_of_the_busier_engine() {
    let engine = decided(PolicySpec::preset("round_robin").expect("a preset"));
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let records = [
        rank(0, Role::Single, 0, 3),
        rank(1, Role::Single, 1, 0),
        behind(2, Role::Single, 2),
    ];
    for seed in 1..=16 {
        let decision = engine.decide(&request(seed), &pool, &records, &cx, &mut scratch);
        assert_eq!(first(&decision), Target::of(&records[2]), "seed {seed}");
    }

    let records = [records[0], records[1], behind(2, Role::Single, 4)];
    for seed in 1..=16 {
        let decision = engine.decide(&request(seed), &pool, &records, &cx, &mut scratch);
        let took = first(&decision);
        assert!(
            took == Target::of(&records[0]) || took == Target::of(&records[1]),
            "a rank of the lighter engine, seed {seed}: {took:?}"
        );
    }
}

/// A rank a filter drops still counts on its engine. One rank of an engine
/// holds twenty requests and is overloaded, so `not-overloaded` drops it,
/// and the engine's other rank is idle. The request goes to the worker alone
/// that holds two, past the idle rank of the engine that holds twenty.
#[test]
fn a_rank_a_filter_drops_still_counts_on_its_engine() {
    let engine = decided(PolicySpec::preset("round_robin").expect("a preset"));
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let busy = rank(0, Role::Single, 0, 20);
    let overloaded = Health {
        overloaded: true,
        ..busy.health
    };
    let records = [
        Candidate {
            health: overloaded,
            ..busy
        },
        rank(1, Role::Single, 1, 0),
        behind(2, Role::Single, 2),
    ];
    for seed in 1..=16 {
        let decision = engine.decide(&request(seed), &pool, &records, &cx, &mut scratch);
        assert_eq!(first(&decision), Target::of(&records[2]), "seed {seed}");
    }
}

/// A pair is decided over two pools made by hand, each side by the
/// program's scorer: the prefill that holds nothing, with the decode that
/// stands alone and holds two, past the idle rank of a decode engine whose
/// other rank holds three. The target is the pair of those two records.
#[test]
fn a_pair_is_decided_over_two_pools_made_by_hand() {
    let pair = PolicySpec::pair_presets("round_robin", "round_robin");
    let engine = decided(pair.expect("two presets"));
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let mut scratch = Scratch::new();
    let prefills = [alone(0, Role::Prefill, 1), alone(1, Role::Prefill, 0)];
    let decodes = [
        rank(0, Role::Decode, 0, 3),
        rank(1, Role::Decode, 1, 0),
        behind(2, Role::Decode, 2),
    ];
    for seed in 1..=16 {
        let pools = PairPools::of(&prefills, &decodes);
        let decision = engine.decide_pair(&request(seed), &pool, pools, &cx, &cx, &mut scratch);
        assert_eq!(
            first(&decision),
            Target::paired(&prefills[1], &decodes[2]),
            "seed {seed}"
        );
    }
}

/// The scorer decides a request the program places, over the list its
/// registry publishes: each worker there is an engine of its own, so a
/// request goes past every worker that has a request under way, to the one
/// that has none.
#[test]
fn the_programs_scorer_places_a_request_past_the_workers_under_way() {
    const PLAIN: Asked<'static> = Asked {
        model: MODEL,
        headers: None,
    };
    let document = settings(PolicySettings::RoundRobin).policy_document();
    let mut document = document.expect("the settings have a document");
    decided_by_engine_load(&mut document);
    let stages = stages().expect("a free name");
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let spec = spec.expect("the catalog holds the scorer");
    let (program, _loads, _task) = started(spec, &workers(3), served);

    let first = program.place(&PLAIN, &[]).expect("a placement");
    let second = program.place(&PLAIN, &[]).expect("a placement");
    let under_way = [first.placement.worker.url(), second.placement.worker.url()];
    assert_ne!(under_way[0], under_way[1], "the second went past the first");
    for turn in 0..8 {
        let answer = program.serve(&PLAIN);
        assert_eq!((answer.status, answer.sent.len()), (200, 1));
        assert!(
            !under_way.contains(&answer.sent[0].as_str()),
            "turn {turn} went to a worker under way: {:?}",
            answer.sent
        );
    }
}
