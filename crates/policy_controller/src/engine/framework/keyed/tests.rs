//! The cases of the keyed decision over rows read one at a time: each of
//! its answers held to the whole decision's over the same rows, the rows it
//! reads counted, and the session table left as the whole decision leaves
//! it.

use std::{cell::Cell, sync::Arc};

use super::{
    super::{Binding, Engine, Filter, Plan, Profile, Scorer, TieBand, WeightedScorer},
    PoolRows,
};
use crate::{
    document::spec::{AppliedSpec, Assignment, Needs, OverloadSpec, PolicySpec, Recovery},
    engine::{
        decision::{Decision, MAX_TARGETS},
        sticky_sessions::{SessionRule, SessionVerdict},
        view::{
            AffinityFacts, Candidate, Constraints, Health, Identity, PoolFacts, RequestFacts,
            Session, WorkerId,
        },
        Scratch, DRAW_BOUND,
    },
    policies::consistent_hashing::{ConsistentHash, HashRing, ListRing, RingPlaces},
    stages::{lifecycle::Lifecycle, pickers::Best},
    tracking::lease::{Lease, Side},
    Draw,
};

/// A pool as a host keeps it for a keyed decision: the rows a whole
/// decision is handed, read one at a time and counted, and the host's ring
/// as the list reads it.
struct Listed<'a> {
    rows: &'a [Candidate],
    ring: &'a ListRing<'a>,
    read: Cell<u32>,
}

/// The address of the worker at `position` of a pool.
fn address(position: u32) -> String {
    format!("http://worker-{position}.pool:8000")
}

/// The host's ring of the workers of the list `rows` are, with the table
/// the list reads it through, as a host makes it once a list.
fn ring_of(rows: &[Candidate]) -> RingPlaces {
    let ring = Arc::new(HashRing::new((0..rows.len() as u32).map(address)));
    let ids: Vec<WorkerId> = rows.iter().map(|row| row.id).collect();
    RingPlaces::new(&ring, &ids)
}

impl<'a> Listed<'a> {
    fn of(rows: &'a [Candidate], ring: &'a ListRing<'a>) -> Self {
        Self {
            rows,
            ring,
            read: Cell::new(0),
        }
    }

    /// The rows read since the last call.
    fn read(&self) -> u32 {
        self.read.take()
    }
}

impl<'a> PoolRows<'a> for Listed<'a> {
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

    fn ring(&self) -> Option<&'a ListRing<'a>> {
        Some(self.ring)
    }
}

/// A pool of `count` rows in an order of its host's own: a row's id is its
/// worker's address's, so the ids rise and fall along it, and its stable
/// key is its id's low half. One row in `down` is not ready and one in
/// `over` is over its thresholds, by `shape`.
fn pool(count: u32, shape: u64, down: usize, over: usize) -> Vec<Candidate> {
    let mut draw = Draw::new(shape);
    (0..count)
        .map(|position| {
            let id = WorkerId::from_address(&address(position));
            // A row's health is drawn behind one draw of its own, which
            // names nothing here.
            draw.next_u64();
            Candidate {
                position,
                id,
                key: id.0 as u32,
                health: Health {
                    healthy: down == 0 || draw.below(down) != 0,
                    overloaded: over != 0 && draw.below(over) == 0,
                    ..Health::default()
                },
                in_flight: draw.below(4) as u32,
                has_capacity: true,
                has_room: true,
                ..Candidate::default()
            }
        })
        .collect()
}

/// The engine of a stock document, as a gateway's start builds it.
fn stock(name: &str) -> AppliedSpec {
    let document = PolicySpec::preset(name).expect("a preset");
    AppliedSpec::validate(&document).expect("a preset validates")
}

/// The stock ring document under a binding that sheds a pool the overload
/// veto empties: a key then keeps no worker over the veto.
fn shedding_ring() -> AppliedSpec {
    let mut document = PolicySpec::preset("consistent_hashing").expect("a preset");
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    AppliedSpec::validate(&document).expect("the document validates")
}

/// A filter of a host's own, which the engine folds into no test of its
/// own: a row is kept where it is ready.
#[derive(Debug)]
struct Ready;

impl Filter for Ready {
    fn name(&self) -> &'static str {
        "ready"
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        row.health.healthy && !facts.tried.contains(&row.id)
    }
}

/// The plain ring under `best` behind a host's own filter, built by hand:
/// no mark of a kept key, and no folded test.
fn hand_built_ring() -> Engine {
    let profile = Profile {
        name: "ring".into(),
        filters: vec![Box::new(Ready)],
        scorers: vec![WeightedScorer {
            scorer: Box::new(ConsistentHash::new(None)),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    };
    Engine::new(vec![profile], vec![single(None)], 1, Needs::default())
}

/// A binding of every model to profile 0 under the single plan.
fn single(sessions: Option<SessionRule>) -> Binding {
    Binding {
        model: "*".into(),
        phases: vec![],
        plan: Plan::Single,
        profile: 0,
        decode_profile: None,
        shadow: None,
        decode_shadow: None,
        sessions,
        overload: None,
    }
}

fn facts_of(pool: &[Candidate]) -> PoolFacts<'static> {
    PoolFacts {
        pool_size: pool.len() as u32,
        generation: 1,
        ..PoolFacts::default()
    }
}

/// What two decisions must agree on: the targets in their order, what the
/// decision did with its key, the veto its trace says was lifted and the
/// profile that decided.
type Answer = (Vec<(u32, WorkerId)>, SessionVerdict, u8, u16);

fn answer(decision: &Decision) -> Option<Answer> {
    let pick = decision.pick()?;
    let targets = pick
        .targets
        .iter()
        .map(|target| (target.position, target.id))
        .collect();
    Some((
        targets,
        pick.session_verdict,
        pick.trace.lifted as u8,
        pick.trace.profile,
    ))
}

/// A ring's keyed request over rows read one at a time is the whole
/// decision's answer: the same targets in the same order and the same mark
/// of a lifted veto, by the stock document, by one whose binding sheds and
/// by an engine built by hand, for a session's key and a client's, where
/// the host says the key stands and where it says nothing, over pools with
/// rows down, over their thresholds, tried, excluded and masked. The whole
/// decision is asked under the ring the keyed one was lent. A decision made
/// here reads no more rows than its walk met.
#[test]
fn a_rings_walk_over_rows_read_one_at_a_time_is_the_whole_decisions_answer() {
    let (stock_ring, shedding) = (stock("consistent_hashing"), shedding_ring());
    let engines: [(&Engine, &str); 3] = [
        (stock_ring.engine(), "stock"),
        (shedding.engine(), "shedding"),
        (&hand_built_ring(), "by hand"),
    ];
    for (engine, name) in engines {
        let (mut decided, mut asked) = (0, 0);
        for (count, shape, down, over) in [
            (1, 1, 0, 0),
            (2, 2, 3, 0),
            (5, 3, 4, 3),
            (8, 4, 7, 4),
            (64, 5, 7, 4),
            (64, 6, 2, 2),
            (64, 7, 0, 1),
            (200, 8, 9, 5),
        ] {
            let rows = pool(count, shape, down, over);
            let places = ring_of(&rows);
            let ring = places.list_ring();
            let listed = Listed::of(&rows, &ring);
            let pool_facts = facts_of(&rows);
            let mut draw = Draw::new(shape ^ 0x5eed);
            let mut scratch = Scratch::new();
            for turn in 0..400_u64 {
                let key = draw.next_u64() | 1;
                // A retry names the workers it tried, the host one it just
                // failed to reach and a row a standing veto of its masks.
                let tried: Vec<WorkerId> = (0..turn % 3)
                    .map(|_| rows[draw.below(rows.len())].id)
                    .collect();
                let excluded = [rows[draw.below(rows.len())].id];
                let stalled = [draw.below(rows.len()) as u32];
                let cx = Constraints {
                    excluded: if turn % 5 == 0 { &excluded } else { &[] },
                    stalled: if turn % 7 == 0 { &stalled } else { &[] },
                    ..Constraints::default()
                };
                let mut facts = RequestFacts {
                    model: "m",
                    session_key: if turn % 4 == 3 {
                        Identity::NONE
                    } else {
                        Identity::hashed(key)
                    },
                    client_key: if turn % 4 >= 2 { key ^ 1 } else { 0 },
                    // Where its host says the key stands, two turns in three.
                    ring_position: if turn % 3 == 2 {
                        0
                    } else {
                        key.rotate_left(23) | 1
                    },
                    tried: &tried,
                    seed: turn,
                    ..RequestFacts::default()
                };
                listed.read();
                let keyed = engine.decide_keyed(&mut facts, &listed, &cx);
                let read = listed.read();
                assert!(
                    facts.ring.is_some(),
                    "{name}: the pool's ring is lent on the way"
                );
                let whole = engine.decide(&facts, &pool_facts, &rows, &cx, &mut scratch);
                asked += 1;
                if let Some(keyed) = keyed {
                    assert_eq!(
                        answer(&keyed),
                        answer(&whole),
                        "{name}, turn {turn} over {count} rows"
                    );
                    assert!(keyed.pick().is_some(), "nothing is shed here");
                    assert!(read < (MAX_TARGETS as u32) + DRAW_BOUND, "{read} rows read");
                    assert_eq!(keyed.trace().survivors.get(), None, "no survivor counted");
                    decided += 1;
                }
            }
        }
        // Most keyed requests are decided by the rows their walk met.
        assert!(
            decided * 2 > asked,
            "{name}: {decided} of {asked} decided by the walk"
        );
    }
}

/// A kept key whose every row the walk met is over its thresholds is the
/// whole decision's: it alone says whether the veto left the pool nobody.
/// With one row in four over, most kept keys are decided by their walk, and
/// with every row over none is.
#[test]
fn a_kept_key_is_decided_by_its_walk_where_a_row_it_met_stands_under_the_veto() {
    let applied = stock("consistent_hashing");
    let engine = applied.engine();
    for (over, least, most) in [(4, 150, 200), (1, 0, 0)] {
        let rows = pool(64, 21, 0, over);
        let places = ring_of(&rows);
        let ring = places.list_ring();
        let listed = Listed::of(&rows, &ring);
        let mut decided = 0;
        for key in 1..=200 {
            let mut facts = RequestFacts {
                model: "m",
                session_key: Identity::hashed(key),
                ..RequestFacts::default()
            };
            let cx = Constraints::default();
            let keyed = engine.decide_keyed(&mut facts, &listed, &cx);
            let whole = engine.decide(&facts, &facts_of(&rows), &rows, &cx, &mut Scratch::new());
            if let Some(keyed) = keyed {
                assert_eq!(answer(&keyed), answer(&whole), "key {key}");
                decided += 1;
            }
        }
        assert!(
            (least..=most).contains(&decided),
            "one row in {over} over: {decided} of 200 decided by the walk"
        );
    }
}

/// A sessions rule's request over rows read one at a time: two engines of
/// one document are sent the same requests, one decides each by its whole
/// pass and the other by the key's pin where the pin stands, and by its
/// whole pass where it does not. Every answer is the same, each commit moves
/// the pins alike, and at the end the two session tables hold every key
/// alike: its pins and the instant it is forgotten. Rows go down and come
/// back, a whole pool goes down for a while, and the clock runs past the
/// rule's idle time. A pin is followed by reading one row or two.
fn pins_followed(whole: &Engine, keyed: &Engine, capped: bool) -> (u32, u32) {
    let mut rows = pool(16, 31, 0, 0);
    // The list's membership stands through the turns, and so does its ring.
    let places = ring_of(&rows);
    let ring = places.list_ring();
    let mut draw = Draw::new(33);
    let keys: Vec<u64> = (0..40).map(|_| draw.next_u64() | 1).collect();
    let (mut scratch, mut keyed_scratch) = (Scratch::new(), Scratch::new());
    let (mut followed, mut asked) = (0, 0);
    let mut now = 1_000_000;
    for turn in 0..3_000_u64 {
        // The clock: seconds mostly, and now and then past four hours.
        now += if turn % 211 == 210 {
            15_000_000_000
        } else {
            1_000_000 * draw.below(90) as u64
        };
        // A row changes its health, and for one stretch every row is down.
        let flipped = draw.below(rows.len());
        rows[flipped].health.healthy = draw.below(4) != 0;
        rows[flipped].health.overloaded = draw.below(5) == 0;
        let outage = (1_200..1_240).contains(&turn);
        let seen: Vec<Candidate> = rows
            .iter()
            .map(|row| Candidate {
                health: Health {
                    healthy: row.health.healthy && !outage,
                    ..row.health
                },
                // The key's requests on the row, as a host counts them.
                session_in_flight: if capped { draw.below(4) as u16 } else { 0 },
                ..*row
            })
            .collect();
        let listed = Listed::of(&seen, &ring);
        let key = keys[draw.below(keys.len())];
        let tried: Vec<WorkerId> = (0..turn % 2)
            .map(|_| seen[draw.below(seen.len())].id)
            .collect();
        // Both decisions read the host's ring where the profile places by
        // one: the whole pass is lent the ring the keyed one is.
        let request = RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            session_counted: true,
            tried: &tried,
            seed: turn,
            now_micros: now,
            ring: Some(&ring),
            ..RequestFacts::default()
        };
        let cx = Constraints::default();
        let pool_facts = facts_of(&seen);
        let by_pass = whole.decide(&request, &pool_facts, &seen, &cx, &mut scratch);
        let mut facts = request;
        let by_pin = keyed.decide_keyed(&mut facts, &listed, &cx);
        asked += 1;
        if by_pin.is_some() {
            followed += 1;
            assert!(listed.read() <= 2, "a pin is one row, its failover another");
        }
        let by_key = match by_pin {
            Some(decision) => decision,
            None => keyed.decide(&facts, &pool_facts, &seen, &cx, &mut keyed_scratch),
        };
        assert_eq!(answer(&by_key), answer(&by_pass), "turn {turn}");
        // The commit: on the first target, or on its alternate now and then.
        if let Some(pick) = by_pass.pick() {
            let at = usize::from(turn % 9 == 8).min(pick.targets.len() - 1);
            let target = pick.targets.as_slice()[at];
            let lease = Lease {
                session: request.session(),
                session_verdict: pick.session_verdict,
                issued_micros: now,
                ..Lease::default()
            };
            whole.on_dispatched(&lease, &target, Side::Single, 0);
            keyed.on_dispatched(&lease, &target, Side::Single, 0);
        }
    }
    for key in keys {
        let session: Session = RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            ..RequestFacts::default()
        }
        .session();
        assert_eq!(
            keyed.sessions().held(session),
            whole.sessions().held(session),
            "key {key:#x}"
        );
    }
    (followed, asked)
}

#[test]
fn a_sessions_pin_is_followed_as_the_whole_decision_follows_it() {
    let (whole, keyed) = (stock("manual"), stock("manual"));
    let (followed, asked) = pins_followed(whole.engine(), keyed.engine(), false);
    assert!(
        followed * 2 > asked,
        "{followed} of {asked} requests followed their pin by one row"
    );
}

/// The stock document's rule with an in-flight cap of two, its key's
/// requests counted on each row: a pin at its cap is the whole decision's
/// to hand on, and a pin under it is followed here.
#[test]
fn a_pin_at_its_rules_cap_is_the_whole_decisions_to_hand_on() {
    let capped = || {
        let mut document = PolicySpec::preset("manual").expect("a preset");
        let rule = document.bindings[0].sessions.as_mut().expect("a rule");
        rule.inflight_cap = 2;
        AppliedSpec::validate(&document).expect("the document validates")
    };
    let (whole, keyed) = (capped(), capped());
    let (followed, asked) = pins_followed(whole.engine(), keyed.engine(), true);
    assert!(
        followed * 5 > asked && followed * 10 < asked * 9,
        "{followed} of {asked} requests followed their pin: the rest were at the cap or held none"
    );
}

/// The stock ring under a sessions rule that leaves a key no pin takes to
/// the ring: such a request keeps its key's worker over the overload veto.
/// Its pin is read with the veto as it stands, and a pin over its
/// thresholds yields to the placement with the veto lifted, which the whole
/// decision makes. A pin that stands is followed here as there.
#[test]
fn a_kept_keys_pin_is_followed_with_the_veto_as_it_stands() {
    let kept = || {
        let mut document = PolicySpec::preset("manual").expect("a preset");
        let ring = PolicySpec::preset("consistent_hashing").expect("a preset");
        document.profiles = ring.profiles;
        let binding = &mut document.bindings[0];
        binding.profile = Some("consistent_hashing".into());
        let rule = binding.sessions.as_mut().expect("a rule");
        rule.assignment = Assignment::Delegate;
        AppliedSpec::validate(&document).expect("the document validates")
    };
    let (whole, keyed) = (kept(), kept());
    let (followed, asked) = pins_followed(whole.engine(), keyed.engine(), false);
    assert!(
        followed * 3 > asked,
        "{followed} of {asked} requests followed their pin by one row"
    );
}

/// What is not decided here is left to the whole decision, and nothing is
/// read or written for it: a request with no key, a request that carries
/// an affinity column, a key with no pin, a document none of whose bindings
/// pins a key or reads a ring.
#[test]
fn a_request_the_key_does_not_decide_is_left_whole_to_the_whole_decision() {
    let rows = pool(16, 41, 0, 0);
    let places = ring_of(&rows);
    let ring = places.list_ring();
    let listed = Listed::of(&rows, &ring);
    let cx = Constraints::default();
    let keyed = RequestFacts {
        model: "m",
        session_key: Identity::hashed(7),
        ..RequestFacts::default()
    };
    let by_ring = stock("consistent_hashing");
    // The ring decides a keyed request, and leaves these.
    assert!(by_ring
        .engine()
        .decide_keyed(&mut { keyed }, &listed, &cx)
        .is_some());
    let column = vec![AffinityFacts::default(); rows.len()];
    for mut left in [
        RequestFacts {
            session_key: Identity::NONE,
            ..keyed
        },
        RequestFacts {
            affinity: &column,
            ..keyed
        },
    ] {
        listed.read();
        assert!(by_ring
            .engine()
            .decide_keyed(&mut left, &listed, &cx)
            .is_none());
        assert_eq!(listed.read(), 0, "no row is read for it");
    }
    // A key the table holds no pin of: the rule's whole decision places
    // and claims it, and nothing is written for it here.
    let manual = stock("manual");
    let mut fresh = RequestFacts {
        now_micros: 5,
        ..keyed
    };
    assert!(manual
        .engine()
        .decide_keyed(&mut fresh, &listed, &cx)
        .is_none());
    assert!(manual.engine().sessions().is_empty(), "no key was written");
    assert!(
        fresh.ring.is_none(),
        "no ring is lent a profile that reads none"
    );
    // A document that pins no key and reads no ring is one test.
    for name in ["round_robin", "least_load", "random", "power_of_two"] {
        let plain = stock(name);
        assert!(plain
            .engine()
            .decide_keyed(&mut { keyed }, &listed, &cx)
            .is_none());
        assert_eq!(listed.read(), 0, "{name}");
    }
    // The prefix ring under its bounded-load rule reads every row's count:
    // its keyed request is lent the ring and left to the whole decision.
    let prefix = stock("prefix_hash");
    let mut bounded = keyed;
    assert!(prefix
        .engine()
        .decide_keyed(&mut bounded, &listed, &cx)
        .is_none());
    assert!(bounded.ring.is_some());
    assert_eq!(listed.read(), 0);
}

/// A scorer that scores every row alike, and states what its pick holds of
/// the request where it is told to.
#[derive(Debug)]
struct Alike {
    states: bool,
}

impl Lifecycle for Alike {}

impl Scorer for Alike {
    fn name(&self) -> &'static str {
        "alike"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand::NORMALIZED
    }

    fn states_overlap(&self) -> bool {
        self.states
    }
}

/// A profile of one scorer under `best`, behind a host's own filter.
fn profile_of(name: &str, scorer: Box<dyn Scorer>) -> Profile {
    Profile {
        name: name.into(),
        filters: vec![Box::new(Ready)],
        scorers: vec![WeightedScorer {
            scorer,
            weight: 1.0,
        }],
        picker: Box::new(Best),
    }
}

/// A binding with a shadow is the whole decision's, which hands the shadow
/// the rows it read. The pool's ring is lent on the way where the binding's
/// profile reads one, and where its shadow's alone does: the two decisions
/// that follow then place by the list's ring.
#[test]
fn a_shadowed_binding_is_lent_the_ring_and_left_to_the_whole_decision() {
    let rows = pool(16, 51, 0, 0);
    let places = ring_of(&rows);
    let ring = places.list_ring();
    let listed = Listed::of(&rows, &ring);
    let cx = Constraints::default();
    let by_ring = || profile_of("ring", Box::new(ConsistentHash::new(None)));
    let plain = || profile_of("plain", Box::new(Alike { states: false }));
    let shadowed = Binding {
        shadow: Some(1),
        ..single(None)
    };
    for (profiles, name) in [
        (vec![by_ring(), plain()], "the ring shadowed"),
        (vec![plain(), by_ring()], "the ring as the shadow"),
    ] {
        let engine = Engine::new(profiles, vec![shadowed.clone()], 1, Needs::default());
        let mut facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(7),
            ..RequestFacts::default()
        };
        assert!(
            engine.decide_keyed(&mut facts, &listed, &cx).is_none(),
            "{name}"
        );
        assert!(facts.ring.is_some(), "{name}: the ring is lent");
        assert_eq!(listed.read(), 0, "{name}: no row is read");
    }
    // With no shadow the same ring profile decides its keyed request here.
    let alone = Engine::new(vec![by_ring()], vec![single(None)], 1, Needs::default());
    let mut facts = RequestFacts {
        model: "m",
        session_key: Identity::hashed(7),
        ..RequestFacts::default()
    };
    assert!(alone.decide_keyed(&mut facts, &listed, &cx).is_some());
}

/// A profile whose scorer states what its pick holds of the request is the
/// whole decision's, which reads the rows for that word: its key's pin is
/// not followed here, where the same rule over a scorer that states nothing
/// follows it by one row.
#[test]
fn a_pin_under_a_scorer_that_states_its_picks_holding_is_left_to_the_whole_decision() {
    let rows = pool(16, 61, 0, 0);
    let places = ring_of(&rows);
    let ring = places.list_ring();
    let listed = Listed::of(&rows, &ring);
    let cx = Constraints::default();
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 0,
        ttl_micros: Some(60_000_000),
        sweep_micros: 60_000_000,
        recovery: Recovery::Original,
    };
    for states in [true, false] {
        let profile = profile_of("alike", Box::new(Alike { states }));
        let engine = Engine::new(vec![profile], vec![single(Some(rule))], 1, Needs::default());
        let request = RequestFacts {
            model: "m",
            session_key: Identity::hashed(9),
            now_micros: 1_000,
            ..RequestFacts::default()
        };
        // The key's first request: no pin stands, and nothing is decided
        // here. The whole decision places and claims it, and its commit
        // makes the pin.
        assert!(engine
            .decide_keyed(&mut { request }, &listed, &cx)
            .is_none());
        let placed = engine.decide(&request, &facts_of(&rows), &rows, &cx, &mut Scratch::new());
        let pick = placed.pick().expect("a pick");
        let lease = Lease {
            session: request.session(),
            session_verdict: pick.session_verdict,
            issued_micros: 1_000,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &pick.targets.as_slice()[0], Side::Single, 0);
        listed.read();
        let followed = engine.decide_keyed(&mut { request }, &listed, &cx);
        if states {
            assert!(followed.is_none(), "the whole decision states the holding");
            assert_eq!(listed.read(), 0);
        } else {
            let followed = followed.expect("the pin is followed");
            let whole = engine.decide(&request, &facts_of(&rows), &rows, &cx, &mut Scratch::new());
            assert_eq!(answer(&followed), answer(&whole));
            assert_eq!(listed.read(), 1, "a pin is one row");
        }
    }
}
