//! The cases of the keyed decision over rows read one at a time: each of
//! its answers held to the whole decision's over the same rows, the rows it
//! reads counted, and the session table left as the whole decision leaves
//! it.

use std::cell::Cell;

use super::{
    super::{Binding, Engine, Filter, Plan, Profile, Scorer, TieBand, WeightedScorer},
    PoolRows,
};
use crate::{
    document::spec::{
        AppliedSpec, Assignment, Needs, OverloadSpec, PolicySpec, Recovery, RulePlaced,
    },
    engine::{
        decision::{Decision, Overlap, MAX_TARGETS},
        sticky_sessions::{SessionRule, SessionVerdict},
        view::{
            AffinityFacts, Candidate, Constraints, Health, Identity, PoolFacts, RequestFacts,
            Session, WorkerId,
        },
        Scratch, DRAW_BOUND,
    },
    policies::consistent_hashing::{ConsistentHash, InOrder},
    stages::{filters::Healthy, lifecycle::Lifecycle, pickers::Best},
    tracking::lease::{Lease, Side},
    Draw,
};

/// A pool as a host keeps it for a keyed decision: the rows a whole
/// decision is handed, read one at a time and counted.
struct Listed<'a> {
    rows: &'a [Candidate],
    read: Cell<u32>,
}

impl<'a> Listed<'a> {
    fn of(rows: &'a [Candidate]) -> Self {
        Self {
            rows,
            read: Cell::new(0),
        }
    }

    /// The rows read since the last call.
    fn read(&self) -> u32 {
        self.read.take()
    }
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

/// A pool of `count` rows in an order of its host's own: ids that rise and
/// fall along it, each row's stable key its id's low half. One row in
/// `down` is not ready and one in `over` is over its thresholds, by `shape`.
fn pool(count: u32, shape: u64, down: usize, over: usize) -> Vec<Candidate> {
    let mut draw = Draw::new(shape);
    (0..count)
        .map(|position| {
            let id = draw.next_u64() | 1;
            Candidate {
                position,
                id: WorkerId(id),
                key: id as u32,
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

/// `rows` as a caller's own list holds them, the row at place `n`
/// registered at `w<n>` and its id that address's, and the caller's ring of
/// them, in an order of its own by `shape`.
fn ringed(rows: &mut [Candidate], shape: u64) -> InOrder {
    let mut order: Vec<String> = rows
        .iter_mut()
        .map(|row| {
            let address = format!("w{}", row.position);
            row.id = WorkerId::from_address(&address);
            address
        })
        .collect();
    let mut draw = Draw::new(shape ^ 0x0a11);
    for at in (1..order.len()).rev() {
        order.swap(at, draw.below(at + 1));
    }
    InOrder(order)
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

/// The plain ring under `best` behind a host's own filter, built by hand
/// at `weight`: no mark of a kept key, and no folded test.
fn hand_built_ring(weight: f32) -> Engine {
    let profile = Profile {
        name: "ring".into(),
        filters: vec![Box::new(Ready)],
        scorers: vec![WeightedScorer {
            scorer: Box::new(ConsistentHash::new(None)),
            weight,
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
/// decision did with its key, the veto its trace says was lifted, the
/// profile that decided and the holding the pick states.
type Answer = (Vec<(u32, WorkerId)>, SessionVerdict, u8, u16, Overlap);

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
        pick.overlap(),
    ))
}

/// The rows a decision's trace names best, each with its score to the bit.
fn best(decision: &Decision) -> Vec<(WorkerId, u32)> {
    let trace = decision.trace();
    trace.top[..usize::from(trace.top_len)]
        .iter()
        .map(|&(id, score)| (id, score.to_bits()))
        .collect()
}

/// A ring's keyed request over rows read one at a time is the whole
/// decision's answer on the ring its caller lent: the same targets in the
/// same order, the same mark of a lifted veto and the same best rows in the
/// trace at the same scores, by the stock document, by one whose binding
/// sheds and by an engine built by hand at a weight whose share of itself
/// is not one, for a session's key and a client's, over pools with rows
/// down, over their thresholds, tried, excluded and masked. A decision made
/// here reads no more rows than its walk met. A kept key whose every row
/// the walk met is over its
/// thresholds is the whole decision's, which alone says whether the veto
/// left the pool nobody: in the pool where every row is over, an answer
/// given here would lack the mark of the lifted veto. A request that is
/// lent no ring is the whole decision's, and so is one its host holds to
/// places of the pool: no row is read for either.
#[test]
fn a_rings_walk_over_rows_read_one_at_a_time_is_the_whole_decisions_answer() {
    let (stock_ring, shedding) = (stock("consistent_hashing"), shedding_ring());
    let engines: [(&Engine, &str); 3] = [
        (stock_ring.engine(), "stock"),
        (shedding.engine(), "shedding"),
        (&hand_built_ring(41.0), "by hand"),
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
            let mut rows = pool(count, shape, down, over);
            let ring = ringed(&mut rows, shape);
            let listed = Listed::of(&rows);
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
                let at = [key];
                let facts = RequestFacts {
                    model: "m",
                    session_key: if turn % 4 == 3 {
                        Identity::NONE
                    } else {
                        Identity::hashed(key)
                    },
                    client_key: if turn % 4 >= 2 { key ^ 1 } else { 0 },
                    tried: &tried,
                    seed: turn,
                    ring: Some(&ring),
                    ring_positions: &at,
                    ..RequestFacts::default()
                };
                listed.read();
                let keyed = engine.decide_keyed(&facts, &listed, &cx);
                let read = listed.read();
                let whole = engine.decide(&facts, &pool_facts, &rows, &cx, &mut scratch);
                asked += 1;
                if let Some(keyed) = keyed {
                    assert_eq!(
                        answer(&keyed),
                        answer(&whole),
                        "{name}, turn {turn} over {count} rows"
                    );
                    assert_eq!(best(&keyed), best(&whole), "{name}, turn {turn}: best");
                    assert!(keyed.pick().is_some(), "nothing is shed here");
                    assert!(read < (MAX_TARGETS as u32) + DRAW_BOUND, "{read} rows read");
                    assert_eq!(keyed.trace().survivors.get(), None, "no survivor counted");
                    decided += 1;
                }
                let unlent = RequestFacts {
                    ring: None,
                    ..facts
                };
                assert!(engine.decide_keyed(&unlent, &listed, &cx).is_none());
                assert_eq!(listed.read(), 0, "{name}: a row read with no ring lent");
                // A request held to places of the pool is the whole
                // decision's, which opens its set with them.
                let places = [(turn % u64::from(count)) as u32];
                let held = Constraints {
                    places: Some(&places),
                    ..cx
                };
                assert!(engine.decide_keyed(&facts, &listed, &held).is_none());
                assert_eq!(listed.read(), 0, "{name}: a row read by its place");
            }
        }
        // Most keyed requests are decided by the rows their walk met.
        assert!(
            decided * 2 > asked,
            "{name}: {decided} of {asked} decided by the walk"
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
        let listed = Listed::of(&seen);
        let key = keys[draw.below(keys.len())];
        let tried: Vec<WorkerId> = (0..turn % 2)
            .map(|_| seen[draw.below(seen.len())].id)
            .collect();
        let request = RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            session_counted: true,
            tried: &tried,
            seed: turn,
            now_micros: now,
            ..RequestFacts::default()
        };
        let cx = Constraints::default();
        let pool_facts = facts_of(&seen);
        let by_pass = whole.decide(&request, &pool_facts, &seen, &cx, &mut scratch);
        let by_pin = keyed.decide_keyed(&request, &listed, &cx);
        asked += 1;
        if by_pin.is_some() {
            followed += 1;
            assert!(listed.read() <= 2, "a pin is one row, its failover another");
        }
        let by_key = match by_pin {
            Some(decision) => decision,
            None => keyed.decide(&request, &pool_facts, &seen, &cx, &mut keyed_scratch),
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
/// read or written for it: a key with no pin, a request with no key, a
/// request that carries an affinity column, a document none of whose
/// bindings pins a key or walks a ring.
#[test]
fn a_request_the_key_does_not_decide_is_left_whole_to_the_whole_decision() {
    let rows = pool(16, 41, 0, 0);
    let listed = Listed::of(&rows);
    let cx = Constraints::default();
    let keyed = RequestFacts {
        model: "m",
        session_key: Identity::hashed(7),
        now_micros: 5,
        ..RequestFacts::default()
    };
    // A key the table holds no pin of: the rule's whole decision places
    // and claims it, and nothing is written for it here.
    let manual = stock("manual");
    let engine = manual.engine();
    assert!(engine.decide_keyed(&keyed, &listed, &cx).is_none());
    assert!(engine.sessions().is_empty(), "no key was written");
    // Its commit makes the pin, and the key's next request follows it
    // here. The same request with no key, or with an affinity column, is
    // left, and no row is read for it.
    let placed = engine.decide(&keyed, &facts_of(&rows), &rows, &cx, &mut Scratch::new());
    let pick = placed.pick().expect("a pick");
    let lease = Lease {
        session: keyed.session(),
        session_verdict: pick.session_verdict,
        issued_micros: 5,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &pick.targets.as_slice()[0], Side::Single, 0);
    assert!(engine.decide_keyed(&keyed, &listed, &cx).is_some());
    let column = vec![AffinityFacts::default(); rows.len()];
    for left in [
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
        assert!(engine.decide_keyed(&left, &listed, &cx).is_none());
        assert_eq!(listed.read(), 0, "no row is read for it");
    }
    // A document that pins no key and walks no ring is one test: no phase
    // of it is answered here. The plain ring's phases are, and its request
    // that is lent no ring is left with no row read. The stock prefix
    // document ranks its ring beside a second scorer, over the whole pool:
    // it is one test too.
    for (name, answered) in [
        ("round_robin", false),
        ("least_load", false),
        ("random", false),
        ("power_of_two", false),
        ("consistent_hashing", true),
        ("prefix_hash", false),
    ] {
        let plain = stock(name);
        let engine = plain.engine();
        assert_eq!(engine.keyed.contains(&true), answered, "{name}");
        assert!(engine.decide_keyed(&keyed, &listed, &cx).is_none());
        assert_eq!(listed.read(), 0, "{name}");
    }
    // A ring at a weight that orders nothing places no request by itself:
    // its whole pass scores every row alike and the picker draws. Nor
    // does one at a weight too small to invert: its share of the sum is
    // not finite, and the whole pass has no score to order by. A weight
    // under the least normal one still places where it inverts.
    for (weight, answered) in [
        (1.0, true),
        (0.0, false),
        (f32::INFINITY, false),
        (1e-39, false),
        (4e-39, true),
    ] {
        let ring = hand_built_ring(weight);
        assert_eq!(ring.keyed.contains(&true), answered, "weight {weight}");
    }
    assert!((1.0f32 / 1e-39).is_infinite() && (1.0f32 / 4e-39).is_finite());
}

/// A scorer that scores every row alike. Where it is told to, it states
/// what its pick holds of the request, it places a keyed request by
/// itself, on the list's first row, and it holds a scope.
#[derive(Debug)]
struct Alike {
    states: bool,
    owns: bool,
    scoped: bool,
}

impl Lifecycle for Alike {}

impl super::super::DecisionScope for Alike {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}
    fn close(&self) {}
}

impl Scorer for Alike {
    fn name(&self) -> &'static str {
        "alike"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand::NORMALIZED
    }

    fn owns(&self) -> bool {
        self.owns
    }

    fn own_by_place(
        &self,
        _: &RequestFacts<'_>,
        _: usize,
        _: &dyn Fn(WorkerId) -> Option<u32>,
        keep: &mut dyn FnMut(u32) -> bool,
    ) -> usize {
        usize::from(self.owns && keep(0))
    }

    fn needs(&self) -> Needs {
        Needs {
            ring: self.owns,
            ..Needs::default()
        }
    }

    fn states_overlap(&self) -> bool {
        self.states
    }

    fn scope(&self) -> Option<&dyn super::super::DecisionScope> {
        self.scoped.then_some(self as _)
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

/// A session's pin is followed by one row whatever the profile's scorers
/// are: under a scorer that states what its pick holds of the request, and
/// under one that states nothing. The whole decision takes a standing pin
/// behind its filters and asks no scorer, and the pick made here is its
/// answer, with no holding stated.
#[test]
fn a_pin_is_followed_under_a_scorer_that_states_its_picks_holding() {
    let rows = pool(16, 61, 0, 0);
    let listed = Listed::of(&rows);
    let cx = Constraints::default();
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 0,
        ttl_micros: Some(60_000_000),
        sweep_micros: 60_000_000,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    for states in [true, false] {
        let stage = Alike {
            states,
            owns: false,
            scoped: false,
        };
        let profile = profile_of("alike", Box::new(stage));
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
        assert!(engine.decide_keyed(&request, &listed, &cx).is_none());
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
        let followed = engine
            .decide_keyed(&request, &listed, &cx)
            .expect("the pin is followed");
        let whole = engine.decide(&request, &facts_of(&rows), &rows, &cx, &mut Scratch::new());
        assert_eq!(answer(&followed), answer(&whole), "states: {states}");
        assert_eq!(listed.read(), 1, "a pin is one row");
        let stated = followed.pick().map(|pick| pick.overlap());
        assert_eq!(stated, Some(Overlap::NONE), "no holding is stated");
    }
}

/// A scorer that could state what its pick holds of the request walks like
/// any other: no entry of the engine asks that word now, so the whole
/// decision would read no row more for it. A keyed request no rule pins is
/// placed here by the one row it reads, whether its scorer states or not.
#[test]
fn a_walk_under_a_scorer_that_states_its_picks_holding_reads_its_one_row() {
    let mut rows = pool(16, 61, 0, 0);
    let ring = ringed(&mut rows, 61);
    let listed = Listed::of(&rows);
    let cx = Constraints::default();
    for states in [true, false] {
        let stage = Alike {
            states,
            owns: true,
            scoped: false,
        };
        let profile = profile_of("alike", Box::new(stage));
        let engine = Engine::new(vec![profile], vec![single(None)], 1, Needs::default());
        let request = RequestFacts {
            model: "m",
            session_key: Identity::hashed(9),
            ring: Some(&ring),
            ring_positions: &[0],
            ..RequestFacts::default()
        };
        let walked = engine.decide_keyed(&request, &listed, &cx);
        let places: Option<Vec<u32>> = walked
            .as_ref()
            .and_then(Decision::pick)
            .map(|pick| pick.targets.iter().map(|target| target.position).collect());
        assert_eq!(places, Some(vec![0]), "states: {states}");
        assert_eq!(listed.read(), 1, "states: {states}");
    }
}

/// A session key under a sessions rule is the rule's on a ring too. With no
/// pin that stands nothing is decided here, though the request's caller
/// lent a ring and says where the key stands on it, and no row is read: the
/// whole decision places the request and claims its key. The same request
/// with no session key is no rule's, and the ring's walk decides it as the
/// whole decision does.
#[test]
fn a_sessions_rules_key_with_no_pin_is_not_walked_on_a_lent_ring() {
    let mut rows = pool(16, 71, 0, 0);
    let ring = ringed(&mut rows, 71);
    let listed = Listed::of(&rows);
    let cx = Constraints::default();
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 0,
        ttl_micros: Some(60_000_000),
        sweep_micros: 60_000_000,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    let profile = profile_of("ring", Box::new(ConsistentHash::new(None)));
    let engine = Engine::new(vec![profile], vec![single(Some(rule))], 1, Needs::default());
    let fresh = RequestFacts {
        model: "m",
        session_key: Identity::hashed(9),
        now_micros: 1_000,
        ring: Some(&ring),
        ring_positions: &[0],
        ..RequestFacts::default()
    };
    assert!(engine.decide_keyed(&fresh, &listed, &cx).is_none());
    assert_eq!(listed.read(), 0, "a row read for a key with no pin");
    let mut scratch = Scratch::new();
    let claimed = engine.decide(&fresh, &facts_of(&rows), &rows, &cx, &mut scratch);
    let verdict = claimed.pick().map(|pick| pick.session_verdict);
    assert!(
        matches!(verdict, Some(SessionVerdict::Claimed(_))),
        "{verdict:?}"
    );
    let unkeyed = RequestFacts {
        session_key: Identity::NONE,
        client_key: 9,
        ..fresh
    };
    let walked = engine.decide_keyed(&unkeyed, &listed, &cx);
    let walked = walked.expect("the ring's walk decides it");
    let whole = engine.decide(&unkeyed, &facts_of(&rows), &rows, &cx, &mut scratch);
    assert_eq!(answer(&walked), answer(&whole));
    assert_eq!(listed.read(), MAX_TARGETS as u32, "the walk's rows alone");
}

/// A standing pin is the whole decision's under a profile one of whose
/// stages opens a scope, and under a binding with a shadow: no row is read
/// here for it. Under neither, the same pin is followed by its one row.
#[test]
fn a_pin_under_a_scope_or_a_shadow_is_left_to_the_whole_decision() {
    let rows = pool(16, 61, 0, 0);
    let listed = Listed::of(&rows);
    let cx = Constraints::default();
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 0,
        ttl_micros: Some(60_000_000),
        sweep_micros: 60_000_000,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    for (scoped, shadow) in [(false, None), (true, None), (false, Some(1))] {
        let alike = || {
            let stage = Alike {
                states: false,
                owns: false,
                scoped,
            };
            let mut profile = profile_of("alike", Box::new(stage));
            // A filter the engine holds itself: a row is asked under a scope too.
            profile.filters = vec![Box::new(Healthy)];
            profile
        };
        let binding = Binding {
            shadow,
            ..single(Some(rule))
        };
        let engine = Engine::new(vec![alike(), alike()], vec![binding], 1, Needs::default());
        let request = RequestFacts {
            model: "m",
            session_key: Identity::hashed(9),
            now_micros: 1_000,
            ..RequestFacts::default()
        };
        let table = engine.sessions();
        table.lookup(request.session(), 1_000, &rule);
        table.dispatched(request.session(), SessionVerdict::Pinned, rows[0].id);
        let followed = engine.decide_keyed(&request, &listed, &cx);
        let left = scoped || shadow.is_some();
        assert_eq!(followed.is_none(), left, "{scoped} {shadow:?}");
        assert_eq!(listed.read(), u32::from(!left), "{scoped} {shadow:?}");
    }
}
