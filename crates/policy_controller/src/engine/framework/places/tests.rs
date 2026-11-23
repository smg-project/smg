//! The cases of a request held to places of its pool, for the passes that
//! hold it: the staged pass and each pass that opens a set of its own hand
//! no row outside the places, under profiles that list no filter and under
//! the crate's presets as they ship.

use std::cell::Cell;

use super::super::{
    Binding, Engine, Picker, Plan, PoolRows, Profile, Scorer, WeightedScorer, DRAW_FROM,
};
use crate::{
    document::spec::{Assignment, Needs, PolicySpec, Recovery, RulePlaced, PRESETS},
    engine::{
        decision::{Decision, LiftedVeto, MAX_TARGETS},
        sticky_sessions::{SessionRule, SessionVerdict},
        view::{
            AffinityHolder, AffinityRows, Candidate, Constraints, Engines, Health, Identity,
            PoolFacts, RequestFacts, WorkerId,
        },
        Scratch,
    },
    policies::{consistent_hashing::ConsistentHash, power_of_two::PowerOfTwo, random::Random},
    stages::{lifecycle::Lifecycle, pickers::Best, scorers::LeastRequests},
    tracking::lease::{Lease, Side},
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

/// A profile of no filter: `scorers` under `picker`.
fn bare(name: &str, scorers: Vec<Box<dyn Scorer>>, picker: Box<dyn Picker>) -> Profile {
    let scorers = scorers
        .into_iter()
        .map(|scorer| WeightedScorer {
            scorer,
            weight: 1.0,
        })
        .collect();
    Profile {
        name: name.into(),
        filters: vec![],
        scorers,
        picker,
    }
}

/// The single plan over the first profile, for every model.
fn single(shadow: Option<u16>) -> Binding {
    Binding {
        model: "*".into(),
        phases: vec![],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow,
        decode_shadow: None,
        sessions: None,
        overload: None,
    }
}

/// Whether the whole pass decided: its trace counts the survivors of its
/// filters, where a draw's and a walk's count none.
fn by_the_pass(decision: &Decision) -> Option<u32> {
    decision.pick()?.trace.survivors.get()
}

/// Every preset, with its filters and with none, over a pool of `rows`
/// rows, for a request with no key: no target stands outside the
/// places, and the presets that place a request by its pool alone place
/// every one. A preset with its filters folds them into one test of a row,
/// asked in one pass (`keep_folded`): that pass steps aside for a request
/// held to places, and the staged pass decides.
fn no_preset_hands_a_target_outside_the_places(rows: u32) {
    let places = [2, 5];
    let by_pool = [
        "random",
        "round_robin",
        "passthrough",
        "power_of_two",
        "least_load",
        "consistent_hashing",
    ];
    let rows = pool(rows);
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

#[test]
fn no_preset_hands_a_target_outside_the_places_under_the_draws_floor() {
    no_preset_hands_a_target_outside_the_places(8);
}

/// Over the draw's floor each pass that draws steps aside for it.
#[test]
fn no_preset_hands_a_target_outside_the_places_over_the_draws_floor() {
    no_preset_hands_a_target_outside_the_places(4 * DRAW_FROM as u32);
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

/// A host states each place once, ascending: a debug build stops at a
/// place stated twice, which the whole pass would name twice.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "the places are ascending, each once")]
fn a_place_stated_twice_stops_a_debug_build() {
    let (engine, rows) = (stock("round_robin", true), pool(8));
    let (cx, mut scratch) = (stating(&[5, 5]), Scratch::new());
    let _ = engine.decide(&facts(1), &of(&rows), &rows, &cx, &mut scratch);
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

/// The draw steps aside: a profile that draws first, with no filter, over
/// a pool above the draw's floor. Held to nothing the decision is the
/// draw's, which counts no survivor. Held to places it is the whole
/// pass's, and no target stands outside them.
#[test]
fn the_draw_steps_aside_for_a_request_held_to_places() {
    let places = [3, 40];
    for name in ["random", "power_of_two"] {
        let engine = stock(name, true);
        let rows = pool(4 * DRAW_FROM as u32);
        for seed in 0..32 {
            let mut scratch = Scratch::new();
            let free = Constraints::default();
            let drawn = engine.decide(&facts(seed), &of(&rows), &rows, &free, &mut scratch);
            assert!(drawn.pick().is_some(), "{name}");
            assert_eq!(by_the_pass(&drawn), None, "{name}: a draw decides");
            let cx = stating(&places);
            let decision = engine.decide(&facts(seed), &of(&rows), &rows, &cx, &mut scratch);
            assert_eq!(by_the_pass(&decision), Some(2), "{name}, seed {seed}");
            let named = named(&decision);
            assert!(
                named.iter().all(|place| places.contains(place)),
                "{name}: {named:?}"
            );
        }
    }
}

/// The draw over rows written when asked steps aside too: handed a
/// request held to places, it draws nothing, every row is written once and
/// the whole pass decides among the places.
#[cfg(feature = "in-process")]
#[test]
fn the_written_draw_steps_aside_for_a_request_held_to_places() {
    let engine = stock("random", true);
    let fixed = pool(4 * DRAW_FROM as u32);
    let places = [3, 40];
    for seed in 0..16 {
        let request = facts(seed);
        let unsaid = of(&fixed);
        let drawn = engine
            .drawn_for(&request, &unsaid, fixed.len())
            .expect("a draw's decision");
        let mut rows = vec![Candidate::default(); fixed.len()];
        let mut written = 0_usize;
        let mut write = |fixed: &Candidate, row: &mut Candidate| {
            written += 1;
            *row = *fixed;
        };
        let (cx, mut scratch) = (stating(&places), Scratch::new());
        let decision = engine.decide_drawn(
            drawn,
            &request,
            &unsaid,
            &fixed,
            &mut rows,
            &mut write,
            &cx,
            &mut scratch,
        );
        assert_eq!(written, fixed.len(), "the whole pass reads every row");
        assert_eq!(by_the_pass(&decision), Some(2), "seed {seed}");
        let named = named(&decision);
        assert!(
            named.iter().all(|place| places.contains(place)),
            "{named:?}"
        );
    }
}

/// A pool of `engines` engines of `ranks` rows each, listed engine after
/// engine: the rows, the list and each engine's end.
fn ranked(engines: u32, ranks: u32) -> (Vec<Candidate>, Vec<u32>, Vec<u32>) {
    let mut rows = pool(engines * ranks);
    for row in &mut rows {
        row.dp_rank = (row.position % ranks) as u16;
        row.dp_size = ranks as u16;
    }
    let listed = (0..engines * ranks).collect();
    let ends = (1..=engines).map(|engine| engine * ranks).collect();
    (rows, listed, ends)
}

/// The draw over a pool's engines steps aside: `power_of_two`, whose
/// picker's unit is the engine and whose scorer answers row by row, with
/// no filter, over a pool that lists its engines. Held to nothing the
/// decision is the draw's over engines. Held to places it is the staged
/// pass's, and no target stands outside them.
#[test]
fn the_draw_over_engines_steps_aside_for_a_request_held_to_places() {
    let engine = stock("power_of_two", true);
    let (rows, listed, ends) = ranked(6, 4);
    let pool = PoolFacts {
        engines: Engines::new(0, &listed, &ends),
        ..of(&rows)
    };
    let places = [1, 13];
    for seed in 0..32 {
        let mut scratch = Scratch::new();
        let free = Constraints::default();
        let drawn = engine.decide(&facts(seed), &pool, &rows, &free, &mut scratch);
        assert!(drawn.pick().is_some());
        assert_eq!(by_the_pass(&drawn), None, "the draw over engines decides");
        let cx = stating(&places);
        let decision = engine.decide(&facts(seed), &pool, &rows, &cx, &mut scratch);
        assert_eq!(by_the_pass(&decision), Some(2), "seed {seed}");
        let named = named(&decision);
        assert!(
            named.iter().all(|place| places.contains(place)),
            "{named:?}"
        );
    }
}

/// A scorer's own walk steps aside: the ring alone under `best`, with no
/// filter, over a pool that lists its engines, for a keyed request. Held
/// to nothing the walk decides. Held to every place but its key's owner,
/// the pass decides, and the request is handed the members the walk names
/// behind the owner, in the walk's order.
#[test]
fn a_scorers_own_walk_steps_aside_for_a_request_held_to_places() {
    let profile = bare(
        "ring",
        vec![Box::new(ConsistentHash::new(None))],
        Box::new(Best),
    );
    let engine = Engine::new(vec![profile], vec![single(None)], 1, Needs::default());
    let (rows, listed, ends) = ranked(8, 2);
    let pool = PoolFacts {
        membership_epoch: 1,
        engines: Engines::new(0, &listed, &ends),
        ..of(&rows)
    };
    for key in 1..=24 {
        let request = RequestFacts {
            session_key: Identity::hashed(key),
            ..facts(key)
        };
        let mut scratch = Scratch::new();
        let free = Constraints::default();
        let walked = engine.decide(&request, &pool, &rows, &free, &mut scratch);
        assert_eq!(by_the_pass(&walked), None, "key {key}: the walk decides");
        let whole = named(&walked);
        assert_eq!(whole.len(), MAX_TARGETS, "key {key}");
        let others: Vec<u32> = (0..rows.len() as u32)
            .filter(|&place| place != whole[0])
            .collect();
        let cx = stating(&others);
        let decision = engine.decide(&request, &pool, &rows, &cx, &mut scratch);
        assert_eq!(
            by_the_pass(&decision),
            Some(others.len() as u32),
            "key {key}"
        );
        let part = named(&decision);
        assert_eq!(part.len(), MAX_TARGETS, "key {key}");
        assert_eq!(part[..MAX_TARGETS - 1], whole[1..], "key {key}");
    }
}

/// A shadow is held to the places of its live decision. Both profiles list
/// no filter, and the shadow's draws first: held to nothing its decision is
/// the draw's. Held to places, neither is handed a row outside them.
#[test]
fn a_shadow_is_held_to_the_places_of_its_live_decision() {
    let profiles = vec![
        bare("live", vec![Box::new(LeastRequests)], Box::new(PowerOfTwo)),
        bare("beside", vec![], Box::new(Random)),
    ];
    let engine = Engine::new(profiles, vec![single(Some(1))], 1, Needs::default());
    let places = [3, 40];
    let rows = pool(4 * DRAW_FROM as u32);
    for seed in 0..32 {
        let (free, mut scratch) = (Constraints::default(), Scratch::new());
        engine.decide_keeping(&facts(seed), &of(&rows), &rows, &free, &mut scratch);
        let drawn = engine
            .shadow(&facts(seed), &of(&rows), &rows, &free, &mut scratch)
            .expect("the binding's shadow");
        assert_eq!(by_the_pass(&drawn), None, "seed {seed}: a draw decides");
        let (cx, mut scratch) = (stating(&places), Scratch::new());
        let live = engine.decide_keeping(&facts(seed), &of(&rows), &rows, &cx, &mut scratch);
        let beside = engine
            .shadow(&facts(seed), &of(&rows), &rows, &cx, &mut scratch)
            .expect("the binding's shadow");
        for decision in [&live, &beside] {
            let named = named(decision);
            assert!(!named.is_empty(), "seed {seed}");
            assert!(
                named.iter().all(|place| places.contains(place)),
                "{named:?}"
            );
        }
    }
}

/// A pool read one row at a time: its rows, and how many were read.
struct Listed<'a> {
    rows: &'a [Candidate],
    read: Cell<u32>,
}

impl PoolRows for Listed<'_> {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn read(&self, place: u32) -> Option<Candidate> {
        self.read.set(self.read.get() + 1);
        self.rows.get(place as usize).copied()
    }

    fn place_of(&self, id: WorkerId) -> Option<u32> {
        let at = self.rows.iter().position(|row| row.id == id)?;
        Some(at as u32)
    }

    fn generation(&self) -> u64 {
        1
    }
}

/// Under the stock ring, with its filters and with none, a keyed request
/// held to every place but its key's owner is handed the members the ring
/// names behind the owner, in the ring's order: as many targets as a
/// request held to nothing, and never the owner.
#[test]
fn a_keyed_request_held_to_places_is_handed_the_rings_next_members() {
    for bare in [false, true] {
        let engine = stock("consistent_hashing", bare);
        let rows = pool(16);
        for key in 1..=40 {
            let request = RequestFacts {
                session_key: Identity::hashed(key),
                ..facts(key)
            };
            let (free, mut scratch) = (Constraints::default(), Scratch::new());
            let whole = named(&engine.decide(&request, &of(&rows), &rows, &free, &mut scratch));
            assert_eq!(whole.len(), MAX_TARGETS, "key {key}");
            let others: Vec<u32> = (0..16).filter(|&place| place != whole[0]).collect();
            let cx = stating(&others);
            let part = named(&engine.decide(&request, &of(&rows), &rows, &cx, &mut scratch));
            assert_eq!(part.len(), MAX_TARGETS, "key {key}, bare {bare}");
            assert_eq!(
                part[..MAX_TARGETS - 1],
                whole[1..],
                "key {key}, bare {bare}"
            );
            assert!(!part.contains(&whole[0]), "key {key}: the owner is handed");
        }
    }
}

/// A profile that reads who holds the prompt, over a pool above the draw's
/// floor, for a request whose record names a holder outside its places.
/// Held to nothing, the holder leads. Held to places, the draw that joins
/// the named rows steps aside and the holder is not handed.
#[test]
fn a_holder_named_outside_the_places_is_not_handed() {
    let engine = stock("cache_aware", false);
    let rows = pool(4 * DRAW_FROM as u32);
    let places = [2, 5];
    let hashes = [0_u64; 16];
    let holders = [AffinityHolder {
        row: 40,
        overlap_blocks: 16,
        predicted_blocks: 0,
    }];
    for seed in 0..24 {
        let request = RequestFacts {
            block_hashes: &hashes,
            prompt_tokens: 256,
            named: Some(AffinityRows {
                holders: &holders,
                warming: &[],
                thin: false,
                mean_in_flight: None,
            }),
            ..facts(seed)
        };
        let mut scratch = Scratch::new();
        let free = Constraints::default();
        let led = engine.decide(&request, &of(&rows), &rows, &free, &mut scratch);
        assert_eq!(
            named(&led).first(),
            Some(&40),
            "seed {seed}: the holder leads"
        );
        let cx = stating(&places);
        let decision = engine.decide(&request, &of(&rows), &rows, &cx, &mut scratch);
        let named = named(&decision);
        assert!(!named.is_empty(), "seed {seed}");
        assert!(
            named.iter().all(|place| places.contains(place)),
            "seed {seed}: {named:?}"
        );
    }
}

/// A pin outside the places stands, and the worker picked is claimed
/// beside it. The keyed decision answers nothing for a request held to
/// places, to one or to none too, and reads no row. Where the places
/// leave out the pin's worker,
/// the rule's recovery places the request among them and claims the worker
/// picked as the key's other pin: behind the standing pin under
/// `original`, ahead of it under `latest`. Where the places keep the pin
/// that leads, the whole decision follows it. A request of the key held to
/// nothing follows that pin by one row's read.
#[test]
fn a_pin_outside_the_places_stands_and_the_worker_picked_is_claimed_beside_it() {
    for recovery in [Recovery::Original, Recovery::Latest] {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 0,
            ttl_micros: Some(60_000_000),
            // Six sweeps an idle time: a key is forgotten within ten
            // seconds of its idle time's end.
            sweep_micros: 10_000_000,
            recovery,
            rule_placed: RulePlaced::Told,
        };
        let profile = bare("random", vec![], Box::new(Random));
        let binding = Binding {
            sessions: Some(rule),
            ..single(None)
        };
        let engine = Engine::new(vec![profile], vec![binding], 1, Needs::default());
        let rows = pool(8);
        let listed = Listed {
            rows: &rows,
            read: Cell::new(0),
        };
        let request = RequestFacts {
            session_key: Identity::hashed(9),
            now_micros: 1_000,
            ..facts(3)
        };
        let key = request.session();
        let free = Constraints::default();
        // The key's first request is placed and claimed, and its dispatch
        // makes the pin.
        let dispatch = |decision: &Decision| {
            let pick = decision.pick().expect("a pick");
            let lease = Lease {
                session: key,
                session_verdict: pick.session_verdict,
                issued_micros: 1_000,
                ..Lease::default()
            };
            let target = pick.targets.as_slice()[0];
            engine.on_dispatched(&lease, &target, Side::Single, 0);
            target
        };
        let mut scratch = Scratch::new();
        let placed = engine.decide(&request, &of(&rows), &rows, &free, &mut scratch);
        let pinned = dispatch(&placed);
        assert_eq!(engine.sessions.pins_at(key, 1_000)[0], pinned.id);
        // Held to every place but the pin's: the pin is not followed, and
        // its row is not read.
        let others: Vec<u32> = (0..8).filter(|&place| place != pinned.position).collect();
        let cx = stating(&others);
        assert!(engine.decide_keyed(&request, &listed, &cx).is_none());
        assert_eq!(listed.read.get(), 0, "{recovery:?}");
        // Held to one place that is not the pin's, and to no place: the
        // keyed decision answers neither, and reads no row.
        let one = [others[0]];
        for places in [&one[..], &[]] {
            let cx = stating(places);
            assert!(engine.decide_keyed(&request, &listed, &cx).is_none());
            assert_eq!(listed.read.get(), 0, "{recovery:?}: {places:?}");
        }
        // The recovery places the request among the places and claims the
        // worker picked beside the standing pin.
        let moved = engine.decide(&request, &of(&rows), &rows, &cx, &mut scratch);
        let pick = moved.pick().expect("a pick");
        let taken = pick.targets.as_slice()[0];
        assert!(others.contains(&taken.position), "{recovery:?}");
        assert_eq!(pick.session_verdict, SessionVerdict::Claimed(taken.id));
        let pins = engine.sessions.pins_at(key, 1_000);
        let stand = match recovery {
            Recovery::Original => [pinned.id, taken.id],
            Recovery::Latest => [taken.id, pinned.id],
        };
        assert_eq!(pins, stand, "{recovery:?}");
        dispatch(&moved);
        assert_eq!(engine.sessions.pins_at(key, 1_000), stand, "{recovery:?}");
        // Held to places that keep the pin that leads, half the key's idle
        // time later: the whole decision's too, which follows the pin and
        // renews it. Unrenewed, the key is forgotten at 70 s: it stands
        // at 80 s.
        let later = RequestFacts {
            now_micros: 30_000_000,
            ..request
        };
        let lead = stand_place(&rows, stand[0]);
        let kept = [lead.min(7 - lead), lead.max(7 - lead)];
        let cx = stating(&kept);
        listed.read.set(0);
        assert!(engine.decide_keyed(&later, &listed, &cx).is_none());
        assert_eq!(listed.read.get(), 0, "{recovery:?}");
        let held = engine.decide(&later, &of(&rows), &rows, &cx, &mut scratch);
        assert_eq!(named(&held), [lead], "{recovery:?}");
        let verdict = held.pick().expect("a pick").session_verdict;
        assert_eq!(verdict, SessionVerdict::Pinned, "{recovery:?}");
        assert_eq!(engine.sessions.pins_at(key, 80_000_000), stand);
        // Held to nothing, the key follows the pin that leads, by one row's
        // read.
        let followed = engine.decide_keyed(&later, &listed, &free);
        assert_eq!(named(&followed.expect("the pin is followed")), [lead]);
        assert_eq!(listed.read.get(), 1, "{recovery:?}");
    }
}

/// The place of the row whose id is `id`.
fn stand_place(rows: &[Candidate], id: WorkerId) -> u32 {
    let row = rows.iter().find(|row| row.id == id);
    row.expect("a row of the pool").position
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
