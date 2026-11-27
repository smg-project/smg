use std::{
    sync::{mpsc, Arc, Mutex, Once},
    time::Duration,
};

use super::*;
use crate::{
    document::spec::{self, PolicySpec, Recovery, RulePlaced},
    engine::{
        decision::{self, Pair},
        sticky_sessions::DecodeVerdict,
        view::{self, Health, Identity, MediaCounts, Role, Session, WorkerId},
    },
    policies::{
        manual::Manual,
        passthrough::Passthrough,
        power_of_two::{PowerOfTwo, PowerOfTwoCountFirst, PowerOfTwoMetric},
        random::Random,
        round_robin::RoundRobin,
    },
    stages::{
        filters::{HasRoom, Healthy, NotOverloaded, NotTried},
        pickers::{self, Best, CountFirst},
        scorers::{self, LeastRequests},
    },
};

mod class_tables;
mod columns_and_named_rows;
mod decode_again;
mod draws;
mod epd_plan;
mod folded_filters;
mod lease_words;
mod pair_sessions;
mod pairs;
mod profile_run;
mod ranks;
mod sessions;
mod shadow_decisions;
mod shadow_reads;
mod sheds;

fn candidate(position: u32, in_flight: u32) -> Candidate {
    Candidate {
        position,
        id: WorkerId(100 + u64::from(position)),
        key: 100 + position,
        health: Health {
            healthy: true,
            ..Health::default()
        },
        in_flight,
        ..Candidate::default()
    }
}

fn least_requests_profile() -> Profile {
    Profile {
        name: "least".into(),
        filters: vec![Box::new(Healthy), Box::new(NotTried)],
        scorers: vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    }
}

fn engine() -> Engine {
    Engine::new(
        vec![least_requests_profile()],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    )
}

fn strict_profile() -> Profile {
    Profile {
        name: "strict".into(),
        filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
        scorers: vec![],
        picker: Box::new(Best),
    }
}

fn strict_engine() -> Engine {
    Engine::new(
        vec![strict_profile()],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    )
}

fn pair_binding() -> Binding {
    Binding {
        model: "*".into(),
        phases: vec![],
        plan: Plan::PdPair,
        profile: 0,
        decode_profile: Some(0),
        shadow: None,
        decode_shadow: None,
        sessions: None,
    }
}

fn decide_pair(engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]) -> Decision {
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    engine.decide_pair(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills,
            decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut scratch,
    )
}

fn single_binding() -> Binding {
    Binding {
        plan: Plan::Single,
        decode_profile: None,
        ..pair_binding()
    }
}

/// A column whose entries hold `overlaps`, in the candidates' order.
fn held(overlaps: &[u32]) -> Vec<AffinityFacts> {
    overlaps
        .iter()
        .map(|&overlap_blocks| AffinityFacts {
            overlap_blocks,
            ..AffinityFacts::default()
        })
        .collect()
}

/// `n` routable candidates of pair class 1.
fn paired(n: u32) -> Vec<Candidate> {
    (0..n)
        .map(|position| Candidate {
            pair_class: 1,
            ..candidate(position, 0)
        })
        .collect()
}

/// The position a decision leads with.
fn lead(decision: &Decision) -> Option<u32> {
    decision
        .pick()
        .and_then(|pick| pick.targets.first().map(|target| target.position))
}

/// A scorer that keeps the rows it was handed and scores a row by its
/// position, the later the better.
#[derive(Debug)]
struct Handed(Arc<Mutex<Vec<u32>>>);

impl Lifecycle for Handed {}

impl Scorer for Handed {
    fn name(&self) -> &'static str {
        "handed"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        assert_eq!(out.len(), kept.len(), "one score a row handed");
        *self.0.lock().expect("the probe's rows") = kept.to_vec();
        for (o, &row) in out.iter_mut().zip(kept) {
            *o = candidates[row as usize].position as f32 / 8.0;
        }
        TieBand::NORMALIZED
    }
}

/// The probe scorer under `picker` over the healthy workers, with the
/// rows the scorer was handed last.
fn probed(
    picker: Box<dyn Picker>,
    sessions: Option<SessionRule>,
) -> (Engine, Arc<Mutex<Vec<u32>>>) {
    let handed = Arc::new(Mutex::new(Vec::new()));
    let engine = Engine::new(
        vec![Profile {
            name: "probed".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![WeightedScorer {
                scorer: Box::new(Handed(Arc::clone(&handed))),
                weight: 1.0,
            }],
            picker,
        }],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions,
        }],
        1,
        Needs::default(),
    );
    (engine, handed)
}

/// The first survivor, unless it is of the refused pair class: then the
/// picker declines.
#[derive(Debug)]
struct RefusesClass(u32);

impl Lifecycle for RefusesClass {}

impl Picker for RefusesClass {
    fn name(&self) -> &'static str {
        "refuses-class"
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let Some(first) = kept.first().map(|&row| &candidates[row as usize]) else {
            return;
        };
        if first.pair_class != self.0 {
            out.push(target_of(first));
        }
    }
}

/// The two slices of a table whose prefill class `k` pairs with the
/// decode classes `lists[k - 1]`.
fn table_of(lists: &[&[u32]]) -> (Vec<u32>, Vec<u32>) {
    let ends = lists
        .iter()
        .scan(0u32, |end, list| {
            *end += list.len() as u32;
            Some(*end)
        })
        .collect();
    (ends, lists.concat())
}

/// Routable idle rows of the given pair classes, in order.
fn classed_rows(classes: &[u32]) -> Vec<Candidate> {
    classes
        .iter()
        .zip(0u32..)
        .map(|(&pair_class, position)| Candidate {
            pair_class,
            ..candidate(position, 0)
        })
        .collect()
}

/// The decode positions a pick's targets name, ascending.
fn decodes_of(decision: &Decision) -> Vec<u32> {
    let mut decodes: Vec<u32> = decision
        .pick()
        .into_iter()
        .flat_map(|pick| pick.targets.iter())
        .filter_map(|target| target.pair.map(|pair| pair.decode_position))
        .collect();
    decodes.sort_unstable();
    decodes
}

/// The reason and the stage of a shed.
fn shed_of(decision: &Decision) -> Option<(ShedReason, &'static str)> {
    match decision {
        Decision::Shed(shed) => Some((shed.reason, shed.emptied_by)),
        _ => None,
    }
}

fn strict_pair_engine() -> Engine {
    Engine::new(
        vec![strict_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    )
}

/// Round robin over the two health tests, with no scorer.
fn round_robin_profile() -> Profile {
    Profile {
        name: "rotated".into(),
        filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
        scorers: vec![],
        picker: Box::new(RoundRobin::new()),
    }
}

/// The engine of one pair profile, on both sides.
fn pair_engine(profile: Profile) -> Engine {
    Engine::new(vec![profile], vec![pair_binding()], 1, Needs::default())
}

/// Hands the first survivor alone: no alternate follows it.
#[derive(Debug)]
struct FirstKept;

impl Lifecycle for FirstKept {}

impl Picker for FirstKept {
    fn name(&self) -> &'static str {
        "first-kept"
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        if let Some(&row) = kept.first() {
            out.push(target_of(&candidates[row as usize]));
        }
    }
}

const SESSIONS: SessionRule = SessionRule {
    assignment: Assignment::Delegate,
    inflight_cap: 2,
    ttl_micros: None,
    sweep_micros: 0,
    recovery: Recovery::Latest,
    rule_placed: RulePlaced::Told,
};

/// The least-requests profile under a sessions rule.
fn session_engine(rule: SessionRule) -> Engine {
    session_engine_of(least_requests_profile(), rule)
}

/// `profile` under a sessions rule.
fn session_engine_of(profile: Profile, rule: SessionRule) -> Engine {
    Engine::new(
        vec![profile],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: Some(rule),
        }],
        1,
        Needs::default(),
    )
}

/// The session `key` names in the keyed cases' requests (model `m`, the
/// single phase): what a lease carries and the table is keyed by.
fn session(key: u64) -> Session {
    RequestFacts {
        model: "m",
        session_key: Identity::hashed(key),
        ..RequestFacts::default()
    }
    .session()
}

/// Decide for `key` at `now` without committing.
fn decide_keyed(engine: &Engine, pool: &[Candidate], key: u64, now: u64) -> Decision {
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::hashed(key),
        now_micros: now,
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    engine.decide(
        &facts,
        &PoolFacts::default(),
        pool,
        &Constraints::default(),
        &mut scratch,
    )
}

/// Decide for `key` at `now` and dispatch the first target, as a host
/// commits it.
fn place_keyed(engine: &Engine, pool: &[Candidate], key: u64, now: u64) -> Target {
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::hashed(key),
        now_micros: now,
        ..RequestFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &facts,
        &PoolFacts::default(),
        pool,
        &Constraints::default(),
        &mut scratch,
    );
    let pick = decision.pick().expect("a pick");
    let target = *pick.targets.first().expect("a target");
    // The lease takes the request's session and the pick's verdict, as a
    // host commits them.
    let lease = Lease {
        session: facts.session(),
        session_verdict: pick.session_verdict,
        issued_micros: now,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &target, Side::Single, 0);
    target
}

/// The `manual` picker's profile: the filters, no scorer.
fn manual_profile() -> Profile {
    Profile {
        name: "manual".into(),
        filters: vec![Box::new(Healthy), Box::new(NotTried)],
        scorers: vec![],
        picker: Box::new(Manual),
    }
}

fn encode_candidate(position: u32, ec_compatible: bool, vision: bool) -> Candidate {
    Candidate {
        role: Role::Encode,
        ec_compatible,
        vision,
        ..candidate(position, 0)
    }
}

fn encode_facts(images: u16) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        phase: Phase::Encode,
        media: MediaCounts {
            images,
            ..MediaCounts::default()
        },
        ..RequestFacts::default()
    }
}

// ---- a decision that draws first

/// `rows` ready candidates in the host's order, the rows `refused`
/// names not ready.
fn pool_of(rows: u32, refused: impl Fn(u32) -> bool) -> Vec<Candidate> {
    (0..rows)
        .map(|i| Candidate {
            health: Health {
                healthy: !refused(i),
                ..Health::default()
            },
            ..candidate(i, i % 5)
        })
        .collect()
}

/// A preset's engine. `power_of_two/requests` is that preset's form
/// that compares its two draws by the requests in flight.
fn preset(name: &str) -> Engine {
    let document = match name {
        "power_of_two/requests" => PolicySpec::power_of_two(PowerOfTwoMetric::Requests),
        name => PolicySpec::preset(name),
    };
    document
        .expect("a preset")
        .validate()
        .expect("the preset validates")
}

fn request(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        seed: Draw::new(seed).next_u64(),
        ..RequestFacts::default()
    }
}

/// The whole pass of the engine's profile over `pool`, as a decision
/// under [`DRAW_FROM`] rows runs it.
fn whole_pass(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Decision {
    let (entry, index, _) = engine
        .resolve(facts.model, Phase::Single)
        .expect("a binding");
    run_whole(
        engine,
        index,
        facts,
        &PoolFacts::default(),
        Set {
            candidates: pool,
            class: None,
            partnered: None,
            fold: None,
            encode: false,
            vetoed: entry.vetoed,
        },
        &Constraints::default(),
        &mut Buffers::default(),
    )
}

fn drawn(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Decision {
    engine.decide(
        facts,
        &PoolFacts::default(),
        pool,
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

fn positions(decision: &Decision) -> Vec<u32> {
    decision.pick().map_or_else(Vec::new, |pick| {
        pick.targets.iter().map(|t| t.position).collect()
    })
}

/// A host's filter around one of the engine's: the same word on
/// every row, and no filter the build knows.
#[derive(Debug)]
struct Hosted(Box<dyn Filter>);
impl Filter for Hosted {
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn soft(&self) -> bool {
        self.0.soft()
    }

    fn row_wise(&self) -> bool {
        self.0.row_wise()
    }

    fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
        self.0.keep(facts, c, cx)
    }
}

/// `release` frees a caught payload whole. Where a payload's drop
/// panics, the payload of that panic is freed the same way, so a storm
/// of such faults leaves none alive. A chain of drops that each raise
/// the next is freed down to the depth, and the one payload past it is
/// forgotten.
#[test]
fn a_storm_of_payloads_whose_drop_panics_leaves_none_alive() {
    use std::{
        panic::resume_unwind,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static ALIVE: AtomicUsize = AtomicUsize::new(0);

    /// A payload counted while it is alive, whose drop raises the next
    /// of its chain `left` times. No hook runs for a payload raised by
    /// `resume_unwind`, so a storm writes nothing.
    struct Chained {
        left: usize,
    }

    impl Chained {
        fn boxed(left: usize) -> Box<Self> {
            ALIVE.fetch_add(1, Ordering::Relaxed);
            Box::new(Self { left })
        }
    }

    impl Drop for Chained {
        fn drop(&mut self) {
            ALIVE.fetch_sub(1, Ordering::Relaxed);
            if self.left > 0 {
                resume_unwind(Self::boxed(self.left - 1));
            }
        }
    }

    for _ in 0..100_000 {
        release(Chained::boxed(1));
    }
    assert_eq!(
        ALIVE.load(Ordering::Relaxed),
        0,
        "payloads alive after a storm of drops that panic"
    );
    // The two chains by their numbers, never by the constant: a depth
    // one off either way fails one of them, and a payload past the
    // depth that is dropped where it should be forgotten fails the
    // second at once, since the last of a chain raises nothing.
    release(Chained::boxed(3));
    assert_eq!(
        ALIVE.load(Ordering::Relaxed),
        0,
        "a chain of the depth's four payloads is freed whole"
    );
    release(Chained::boxed(4));
    assert_eq!(
        ALIVE.load(Ordering::Relaxed),
        1,
        "a chain one past the depth ends there, with one payload forgotten"
    );
}
