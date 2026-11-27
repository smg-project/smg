use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex, OnceLock, Weak,
};

use crate::{
    document::spec::{
        AppliedSpec, Assignment, Needs, OverloadSpec, PolicySpec, PrefixAffinityParams, Recovery,
        RulePlaced,
    },
    engine::{
        decision::{Decision, LiftedVeto, ShedReason, Survivors, Targets},
        framework::{
            Binding, DecisionScope, Engine, Filter, FirstPartners, Marks, PairPools, PairSide,
            PairTable, Picker, Plan, Profile, Scorer, Scratch, TieBand, WeightedScorer, DRAW_FROM,
            PARTNERED, WRITTEN_ROWS,
        },
        sticky_sessions::{SessionRule, SessionVerdict},
        view::{
            AffinityFacts, AffinityHolder, AffinityRows, Candidate, Constraints, Engines, Health,
            Identity, Phase, PoolFacts, RequestFacts, WorkerId,
        },
    },
    policies::{
        cache_aware::{Gates, PrefixAffinity},
        consistent_hashing::ConsistentHash,
        manual::Manual,
    },
    stages::{
        filters::{HasCapacity, HasRoom, Healthy, NotOverloaded, NotTried},
        lifecycle::Lifecycle,
        pickers::Best,
        registry::StageRegistry,
    },
    tracking::lease::{Lease, Side},
};

mod keyed_sessions;
mod liveness;
mod overload;
mod pair;
mod pair_lift;
mod pair_shadows;

/// A ready worker over its thresholds, holding `in_flight` requests.
fn over(position: u32, in_flight: u32) -> Candidate {
    Candidate {
        position,
        id: WorkerId(100 + u64::from(position)),
        key: 100 + position,
        health: Health {
            healthy: true,
            overloaded: true,
            ..Health::default()
        },
        in_flight,
        ..Candidate::default()
    }
}

/// The engine of one preset; `shed` writes the binding's
/// `overload.shed_when_all_vetoed`.
fn engine_of(preset: &str, shed: bool) -> Engine {
    let mut spec = PolicySpec::preset(preset).expect("the preset's form");
    if shed {
        spec.bindings
            .first_mut()
            .expect("the preset's binding")
            .overload = Some(OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        });
    }
    spec.validate().expect("the preset builds")
}

/// The round robin preset with its one profile's hard filters replaced.
fn engine_filtering(filters: &[&str], shed: bool) -> Engine {
    let mut spec = PolicySpec::preset("round_robin").expect("the preset's form");
    for profile in spec.profiles.values_mut() {
        profile.filter = filters.iter().map(|&name| name.into()).collect();
    }
    if shed {
        spec.bindings
            .first_mut()
            .expect("the preset's binding")
            .overload = Some(OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        });
    }
    spec.validate().expect("the document builds")
}

/// A request of the cases' host, which writes a key's counts into its
/// rows.
fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        seed,
        session_counted: true,
        ..RequestFacts::default()
    }
}

fn decide(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Decision {
    engine.decide(
        facts,
        &PoolFacts::default(),
        pool,
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

/// Refuses every row and counts the rows it was asked about.
#[derive(Debug)]
struct Refuses(Arc<AtomicUsize>);
impl Filter for Refuses {
    fn name(&self) -> &'static str {
        "refuses"
    }
    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        false
    }
}

/// The positions a pick names, best first.
fn order(decision: &Decision) -> Vec<u32> {
    let pick = decision.pick().expect("a pick");
    pick.targets.iter().map(|t| t.position).collect()
}

/// The reason and the stage of a shed.
fn shed_of(decision: &Decision) -> (ShedReason, &'static str) {
    match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("not a shed: {other:?}"),
    }
}

/// [`shed_of`], and the side of a pair the shed names.
fn sided(decision: &Decision) -> ((ShedReason, &'static str), Option<PairSide>) {
    match decision {
        Decision::Shed(shed) => ((shed.reason, shed.emptied_by), shed.side),
        other => panic!("not a shed: {other:?}"),
    }
}

/// The pair plan's shed over `prefills` and one free decode. The prefill
/// side is filtered by `veto` and then by a filter that refuses every
/// row, under the host's mask over the prefills and the rows of it the
/// host holds (`masks`). With the shed, the count of the prefills that
/// filter was asked about.
fn pairs_side_shed(
    veto: Box<dyn Filter>,
    prefills: [Candidate; 2],
    masks: (&[u32], &[u32]),
) -> ((ShedReason, &'static str), usize) {
    let asked = Arc::new(AtomicUsize::new(0));
    let prefill_side = Profile {
        name: "counted".into(),
        filters: vec![veto, Box::new(Refuses(Arc::clone(&asked)))],
        scorers: vec![],
        picker: Box::new(Best),
    };
    let decode_side = Profile {
        name: "plain".into(),
        filters: vec![Box::new(NotOverloaded)],
        scorers: vec![],
        picker: Box::new(Best),
    };
    let engine = Engine::new(
        vec![prefill_side, decode_side],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(1),
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    let paired = |mut worker: Candidate| {
        worker.pair_class = 1;
        worker
    };
    let prefills = prefills.map(paired);
    let mut free = paired(over(0, 0));
    free.health.overloaded = false;
    let decodes = [free];
    let decision = engine.decide_pair(
        &facts(1),
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints {
            stalled: masks.0,
            held: masks.1,
            ..Constraints::default()
        },
        &Constraints::default(),
        &mut Scratch::new(),
    );
    (shed_of(&decision), asked.load(Ordering::Relaxed))
}

/// A pair's engine built by hand: each side filters by the health stage
/// and the overload veto and takes the best of equals, so a side the
/// rule steers is the only one whose pick the requests in flight
/// decide. `shed` makes its binding shed a pool the veto empties. Each
/// side's shadow is a twin of its profile.
fn pair_engine(shed: bool) -> Engine {
    pair_engine_of(shed, [vetoes(), vetoes()])
}

/// The health stage and the overload veto: a side's filters.
fn vetoes() -> Vec<Box<dyn Filter>> {
    vec![Box::new(Healthy), Box::new(NotOverloaded)]
}

/// [`pair_engine`] with its live sides' filters given: the prefill
/// side's, then the decode side's.
fn pair_engine_of(shed: bool, [prefill, decode]: [Vec<Box<dyn Filter>>; 2]) -> Engine {
    pair_engine_holding(shed, [prefill, decode], [None, None])
}

/// A scorer that stands in for a stage that prefers the worker that
/// holds a session's prefix: the row at `holder` scores highest. Two
/// words of the stage are apart here. `reads` says whether it declares
/// that it reads affinity ([`Needs::affinity`]), and `weighs` whether
/// it says that it weighs load against affinity by a gate of its own
/// ([`Scorer::balances_load_against_affinity`]).
#[derive(Clone, Copy, Debug)]
struct Holds {
    holder: u32,
    reads: bool,
    weighs: bool,
}

impl Lifecycle for Holds {}

impl Scorer for Holds {
    fn name(&self) -> &'static str {
        "holds"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (score, &row) in out.iter_mut().zip(kept) {
            let holds = candidates[row as usize].position == self.holder;
            *score = if holds { 1.0 } else { 0.0 };
        }
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            affinity: self.reads,
            ..Needs::default()
        }
    }

    fn balances_load_against_affinity(&self) -> bool {
        self.weighs
    }
}

/// [`pair_engine_of`] with a scorer on the prefill side: the live
/// prefill profile's, then its shadow's. `None` leaves that profile
/// with no scorer.
fn pair_engine_holding(
    shed: bool,
    [prefill, decode]: [Vec<Box<dyn Filter>>; 2],
    holds: [Option<Holds>; 2],
) -> Engine {
    pair_engine_shadowed(shed, [prefill, decode, vetoes()], holds)
}

/// [`pair_engine_holding`] with the prefill shadow's own filters: the
/// live prefill side's, the decode side's, then that shadow's.
fn pair_engine_shadowed(
    shed: bool,
    [prefill, decode, twin]: [Vec<Box<dyn Filter>>; 3],
    [live, shadow]: [Option<Holds>; 2],
) -> Engine {
    let side = |name: &str, filters, holds: Option<Holds>| Profile {
        name: name.into(),
        filters,
        scorers: holds
            .map(|holds| WeightedScorer {
                scorer: Box::new(holds),
                weight: 1.0,
            })
            .into_iter()
            .collect(),
        picker: Box::new(Best),
    };
    let mut engine = Engine::new(
        vec![
            side("prefill", prefill, live),
            side("decode", decode, None),
            side("twin", vetoes(), None),
            side("prefill-twin", twin, shadow),
        ],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(1),
            shadow: Some(3),
            decode_shadow: Some(2),
            sessions: None,
        }],
        1,
        Needs::default(),
    );
    if shed {
        engine.shed_when_all_vetoed(0);
    }
    engine
}

/// `worker` in the pair class `class`: classes pair by their number.
fn classed(mut worker: Candidate, class: u32) -> Candidate {
    worker.pair_class = class;
    worker
}

/// The pair `engine` decides over `prefills` and `decodes`.
fn decide_pair(engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]) -> Decision {
    let pools = PairPools {
        prefills,
        decodes,
        ..PairPools::default()
    };
    engine.decide_pair(
        &facts(1),
        &PoolFacts::default(),
        pools,
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

/// The pair a pick names first: its prefill's position, then its decode's.
fn pair_of(decision: &Decision) -> (u32, u32) {
    let pick = decision.pick().expect("a pick");
    let first = pick.targets.first().expect("a target");
    let decode = first.pair.expect("a pair's target").decode_position;
    (first.position, decode)
}

/// The pools of a fleet some of whose partners go first. Both prefill
/// classes pair with both decode classes, and prefill class 1 is paired
/// with decode class 1 first. Prefill class 2 has no first partner. A
/// gateway hands such classes for the workers of one known runtime
/// beside workers whose runtime is unknown.
fn first_pools<'a>(prefills: &'a [Candidate], decodes: &'a [Candidate]) -> PairPools<'a> {
    PairPools {
        prefills,
        decodes,
        table: PairTable::new(&[2, 4], &[1, 2, 1, 2]),
        first: FirstPartners::new(&[1, 1], &[1]),
        ..PairPools::default()
    }
}

/// An engine of one profile: `veto` and a filter that refuses every row,
/// the veto listed first when `veto_first`. With it, the count of the
/// rows the refusing filter was asked about.
fn counted(veto: Box<dyn Filter>, veto_first: bool) -> (Engine, Arc<AtomicUsize>) {
    counted_among(|refuses| {
        if veto_first {
            vec![veto, refuses]
        } else {
            vec![refuses, veto]
        }
    })
}

/// An engine of one profile under the filters `listed` gives, among
/// them a filter that refuses every row. With it, the count of the rows
/// that filter was asked about.
fn counted_among(
    listed: impl FnOnce(Box<dyn Filter>) -> Vec<Box<dyn Filter>>,
) -> (Engine, Arc<AtomicUsize>) {
    let asked = Arc::new(AtomicUsize::new(0));
    let filters = listed(Box::new(Refuses(Arc::clone(&asked))));
    let engine = Engine::new(
        vec![Profile {
            name: "counted".into(),
            filters,
            scorers: vec![],
            picker: Box::new(Best),
        }],
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
    );
    (engine, asked)
}

/// A ready worker under its thresholds, holding `in_flight` requests.
fn free(position: u32, in_flight: u32) -> Candidate {
    let mut worker = over(position, in_flight);
    worker.health.overloaded = false;
    worker
}

fn keyed(key: &'static str) -> RequestFacts<'static> {
    RequestFacts {
        session_key: Identity::plain(key),
        ..facts(1)
    }
}

/// The ring's order for the request's key over `pool`, found by probes
/// and never written as places: the pick with every worker ready and
/// under its thresholds, then with the members found so far not ready.
fn ring_order(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Vec<u32> {
    let mut probe = pool.to_vec();
    for worker in &mut probe {
        worker.health.overloaded = false;
    }
    let mut walk = Vec::with_capacity(pool.len());
    while walk.len() < pool.len() {
        let member = order(&decide(engine, facts, &probe))[0];
        walk.push(member);
        probe[member as usize].health.healthy = false;
    }
    walk
}

/// The ring's profile under the manual preset's sessions rule, with the
/// header that rule reads admitted by the view. No gateway composes the
/// two: the composition is this engine's own.
fn ring_under_a_rule(assignment: Assignment) -> Engine {
    ring_under(assignment, None)
}

/// The same composition under a rule that delegates, with the rule's
/// in-flight cap set: the manual preset's own rule has none.
fn ring_under_a_cap(cap: u16) -> Engine {
    ring_under(Assignment::Delegate, Some(cap))
}

fn ring_under(assignment: Assignment, cap: Option<u16>) -> Engine {
    let mut spec = PolicySpec::preset("consistent_hashing").expect("the preset's form");
    let manual = PolicySpec::preset("manual").expect("the preset's form");
    let mut rule = manual
        .bindings
        .first()
        .and_then(|binding| binding.sessions.clone())
        .expect("the manual preset's rule");
    rule.assignment = assignment;
    if let Some(cap) = cap {
        rule.inflight_cap = cap;
    }
    spec.view.headers = manual.view.headers.clone();
    let binding = spec.bindings.first_mut().expect("the preset's binding");
    // The rule names the key's header, so the ring's own sources go: a
    // binding names them once.
    binding.routing_key = None;
    binding.sessions = Some(rule);
    spec.validate().expect("the document builds")
}

/// The worker a request keyed `k1` is sent to, dispatched there so its
/// pin is written.
fn dispatched(engine: &Engine, pool: &[Candidate], now: u64) -> u32 {
    let request = RequestFacts {
        session_key: Identity::plain("k1"),
        now_micros: now,
        ..facts(now)
    };
    let decision = decide(engine, &request, pool);
    let pick = *decision.pick().expect("a pick");
    let target = *pick.targets.first().expect("a target");
    let lease = Lease {
        issued_micros: now,
        session: request.session(),
        session_verdict: pick.session_verdict,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &target, Side::Single, 0);
    target.position
}

/// Counts a scope's opens and closes.
#[derive(Debug, Default)]
struct Scope {
    opened: AtomicUsize,
    closed: AtomicUsize,
}

impl DecisionScope for Scope {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        self.opened.fetch_add(1, Ordering::Relaxed);
    }
    fn close(&self) {
        self.closed.fetch_add(1, Ordering::Relaxed);
    }
}

/// A concurrent decision, staged: the engine it runs on and the pool it
/// sees.
#[derive(Debug, Default)]
struct Gap {
    engine: OnceLock<Weak<Engine>>,
    staged: Mutex<Vec<Candidate>>,
}

/// Keeps every row. The first time it is asked under the lifted veto it
/// runs the request once more, on the same engine and over the staged
/// pool: another decision's whole run, in the gap between a decision's
/// read of the pins and its own claim.
#[derive(Debug)]
struct InTheGap(Arc<Gap>);

impl Filter for InTheGap {
    fn name(&self) -> &'static str {
        "in_the_gap"
    }
    fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, cx: &Constraints<'_>) -> bool {
        if cx.overload_lifted {
            let staged = std::mem::take(&mut *self.0.staged.lock().expect("the staged pool"));
            if let Some(engine) = self.0.engine.get().and_then(Weak::upgrade) {
                if !staged.is_empty() {
                    let mut scratch = Scratch::new();
                    let _ = engine.decide(
                        facts,
                        &PoolFacts::default(),
                        &staged,
                        &Constraints::default(),
                        &mut scratch,
                    );
                }
            }
        }
        true
    }
}

/// The ring's profile built by hand with `last` behind the health stage
/// and the overload veto, marked as the validator marks the ring's
/// profile, under `rule`.
fn ring_behind(last: Box<dyn Filter>, rule: Option<SessionRule>) -> Engine {
    let mut engine = Engine::new(
        vec![Profile {
            name: "ring".into(),
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded), last],
            scorers: vec![WeightedScorer {
                scorer: Box::new(ConsistentHash::new(None)),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        }],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: rule,
        }],
        1,
        Needs::default(),
    );
    engine.places_by_key(0);
    engine
}

/// [`ring_behind`] with [`InTheGap`] as that filter.
fn ring_with_a_gap(rule: Option<SessionRule>) -> (Arc<Engine>, Arc<Gap>) {
    let gap = Arc::new(Gap::default());
    let engine = Arc::new(ring_behind(Box::new(InTheGap(Arc::clone(&gap))), rule));
    gap.engine
        .set(Arc::downgrade(&engine))
        .expect("the engine is set once");
    (engine, gap)
}

/// The claim's lead reads the survivors the placement ran over. Another
/// decision of the key, which sees the key's worker down, claims the
/// ring's next worker between this decision's read of the pins and its
/// own claim. This decision sees that worker over its thresholds: the
/// lifted pass keeps it, and the decision follows the claim onto it.
/// Read over the workers the standing veto kept, the lead would not
/// find the claimed worker, and the two decisions would part.
#[test]
fn a_claim_made_in_the_gap_is_followed_onto_a_worker_the_lift_keeps() {
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 0,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    let (ringed, _) = ring_with_a_gap(None);
    let (engine, gap) = ring_with_a_gap(Some(rule));
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&ringed, &keyed("k1"), &pool);
    let mut theirs = pool.to_vec();
    theirs[ring[0] as usize].health.healthy = false;
    *gap.staged.lock().expect("the staged pool") = theirs;
    pool[ring[1] as usize].health.overloaded = true;
    let decision = decide(&engine, &keyed("k1"), &pool);
    assert!(
        gap.staged.lock().expect("the staged pool").is_empty(),
        "the other decision ran"
    );
    assert_eq!(order(&decision)[..2], [ring[1], ring[0]]);
    assert_eq!(
        decision.pick().expect("a pick").session_verdict,
        SessionVerdict::Pinned,
        "the claim of record is the other decision's"
    );
}

/// A followed claim leads no worker that carries the cap's requests of
/// the key either. This decision's own placement, the ring's first
/// worker, carries them: it leaves the targets behind the claimed
/// worker, where a dispatch would write it among the key's pins.
#[test]
fn a_followed_claim_names_no_alternate_at_the_rules_cap() {
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    let (ringed, _) = ring_with_a_gap(None);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&ringed, &keyed("k1"), &pool);
    let (engine, gap) = ring_with_a_gap(Some(rule));
    // The other decision sees the ring's first worker down and claims
    // the ring's second.
    let mut theirs = pool.to_vec();
    theirs[ring[0] as usize].health.healthy = false;
    *gap.staged.lock().expect("the staged pool") = theirs;
    pool[ring[0] as usize].session_in_flight = 2;
    let decision = decide(&engine, &keyed("k1"), &pool);
    assert!(
        gap.staged.lock().expect("the staged pool").is_empty(),
        "the other decision ran"
    );
    let pick = decision.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
    assert_eq!(
        order(&decision),
        [ring[1], ring[2], ring[3]],
        "the free alternates alone"
    );
}

/// A claim made in the gap beside a pin that stands is followed too,
/// and the pin stays. The key's pin is the ring's third worker, over
/// its thresholds in both views: each decision's lifted pass keeps it.
/// Another decision of the key, which sees the ring's first worker
/// down, claims the ring's next worker between this decision's read of
/// the pins and its own claim. This decision's claim sees the newcomer
/// beside the pin it read and follows the other's worker, under either
/// recovery. A second claim there would drop the pin.
#[test]
fn a_claim_made_in_the_gap_beside_a_standing_pin_is_followed() {
    for recovery in [Recovery::Original, Recovery::Latest] {
        let rule = SessionRule {
            assignment: Assignment::Delegate,
            inflight_cap: 0,
            ttl_micros: None,
            sweep_micros: 0,
            recovery,
            rule_placed: RulePlaced::Told,
        };
        let (ringed, _) = ring_with_a_gap(None);
        let (engine, gap) = ring_with_a_gap(Some(rule));
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let ring = ring_order(&ringed, &keyed("k1"), &pool);
        let request = keyed("k1");
        // The key's pin is the ring's third worker, over its thresholds
        // in both views: each decision's lifted pass keeps it.
        let pinned = ring[2] as usize;
        let table = engine.sessions();
        table.lookup(request.session(), request.now_micros, &rule);
        table.dispatched(
            request.session(),
            SessionVerdict::Pinned,
            pool[pinned].id,
            0,
        );
        pool[pinned].health.overloaded = true;
        let mut theirs = pool.to_vec();
        theirs[ring[0] as usize].health.healthy = false;
        *gap.staged.lock().expect("the staged pool") = theirs;
        let decision = decide(&engine, &request, &pool);
        assert!(
            gap.staged.lock().expect("the staged pool").is_empty(),
            "the other decision ran"
        );
        assert_eq!(order(&decision)[..2], [ring[1], ring[0]], "{recovery:?}");
        assert_eq!(
            decision.pick().expect("a pick").session_verdict,
            SessionVerdict::Pinned,
            "{recovery:?}: the claim of record is the other decision's"
        );
        let pins = engine.sessions().pins(request.session()).expect("the pins");
        assert!(pins.contains(&pool[pinned].id), "{recovery:?}: {pins:?}");
        assert!(
            pins.contains(&pool[ring[1] as usize].id),
            "{recovery:?}: {pins:?}"
        );
    }
}

/// The ring's scorer. The first time it is asked it runs the request
/// once more, on the same engine and over the staged pool: another
/// decision's whole run between a decision's read of the pins and its
/// own claim, under a profile that lifts no veto.
#[derive(Debug)]
struct ScoresInTheGap(Arc<Gap>, ConsistentHash);

impl Lifecycle for ScoresInTheGap {}

impl Scorer for ScoresInTheGap {
    fn name(&self) -> &'static str {
        "scores_in_the_gap"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let staged = std::mem::take(&mut *self.0.staged.lock().expect("the staged pool"));
        if let Some(engine) = self.0.engine.get().and_then(Weak::upgrade) {
            if !staged.is_empty() {
                let mut scratch = Scratch::new();
                let _ = engine.decide(
                    facts,
                    &PoolFacts::default(),
                    &staged,
                    &Constraints::default(),
                    &mut scratch,
                );
            }
        }
        self.1.score(facts, candidates, kept, out)
    }
}

/// The ring's stages built by hand under `rule`, with [`ScoresInTheGap`]
/// as the scorer and no mark of the validator's: a keyed request of
/// this profile meets the veto as it stands.
fn plain_ring_with_a_gap(rule: SessionRule) -> (Arc<Engine>, Arc<Gap>) {
    let gap = Arc::new(Gap::default());
    let engine = Arc::new(Engine::new(
        vec![Profile {
            name: "ring".into(),
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            scorers: vec![WeightedScorer {
                scorer: Box::new(ScoresInTheGap(Arc::clone(&gap), ConsistentHash::new(None))),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        }],
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
    ));
    gap.engine
        .set(Arc::downgrade(&engine))
        .expect("the engine is set once");
    (engine, gap)
}

/// One decision of the key `k1` with another's in its gap, the pair on
/// an engine of its own: the other sees the ring's first worker down
/// and claims the ring's second. `shape` sets this decision's view of
/// four free workers. The lead, the verdict and the trace's lift.
fn followed(
    (engine, gap): (Arc<Engine>, Arc<Gap>),
    ring: &[u32],
    shape: impl Fn(&mut [Candidate]),
) -> (u32, SessionVerdict, LiftedVeto) {
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let mut theirs = pool.to_vec();
    theirs[ring[0] as usize].health.healthy = false;
    *gap.staged.lock().expect("the staged pool") = theirs;
    shape(&mut pool);
    let decision = decide(&engine, &keyed("k1"), &pool);
    assert!(
        gap.staged.lock().expect("the staged pool").is_empty(),
        "the other decision ran"
    );
    let verdict = decision.pick().expect("a pick").session_verdict;
    (order(&decision)[0], verdict, decision.trace().lifted)
}

/// Another request's claim is a pin of the key: the rule's in-flight
/// cap holds for it as for the pin read first, whether the decision
/// placed the key on that worker or follows the claim onto it. Under
/// the cap the decision follows the claim onto a worker over its
/// thresholds.
/// Where that worker carries the cap's requests of the key, the request
/// is put to the ring again by the veto as it stands. A survivor under
/// the cap takes the key. A survivor at the cap takes the request and
/// no pin moves, though the lifted pass would keep a worker under the
/// cap. Placed on the claimed worker again, the key stays on the claim.
#[test]
fn a_followed_claim_at_the_rules_cap_hands_the_key_on() {
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    let (ringed, _) = ring_with_a_gap(None);
    let pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&ringed, &keyed("k1"), &pool);
    let (claimed, last) = (ring[1] as usize, ring[3] as usize);
    // This decision sees the claimed worker over its thresholds, with
    // `held` requests of the key: the lifted pass alone keeps it.
    let at = |held: u16| {
        move |pool: &mut [Candidate]| {
            pool[claimed].health.overloaded = true;
            pool[claimed].session_in_flight = held;
        }
    };
    let (pinned, handed) = (SessionVerdict::Pinned, SessionVerdict::Respill);
    let none = SessionVerdict::None;
    let under = followed(ring_with_a_gap(Some(rule)), &ring, at(1));
    assert_eq!(under, (ring[1], pinned, LiftedVeto::None), "under the cap");
    let on = followed(ring_with_a_gap(Some(rule)), &ring, at(2));
    assert_eq!(on, (ring[0], handed, LiftedVeto::None), "at the cap");
    // Every worker the standing veto keeps carries the cap as well.
    // The ring's last, over its thresholds, is under it.
    let lifted = followed(ring_with_a_gap(Some(rule)), &ring, |pool| {
        for candidate in pool.iter_mut() {
            candidate.session_in_flight = 2;
        }
        pool[claimed].health.overloaded = true;
        pool[last].health.overloaded = true;
        pool[last].session_in_flight = 0;
    });
    assert_eq!(
        lifted,
        (ring[0], none, LiftedVeto::None),
        "no veto is lifted"
    );
    let stays = followed(ring_with_a_gap(Some(rule)), &ring, |pool| {
        for candidate in pool.iter_mut() {
            candidate.session_in_flight = 2;
        }
        pool[claimed].health.overloaded = true;
    });
    assert_eq!(
        stays,
        (ring[0], none, LiftedVeto::None),
        "no worker under it"
    );
    // This decision sees the ring's first worker down as the other
    // does: its own placement is the claimed worker.
    let same = followed(ring_with_a_gap(Some(rule)), &ring, |pool| {
        pool[ring[0] as usize].health.healthy = false;
        pool[claimed].session_in_flight = 2;
    });
    assert_eq!(
        same,
        (ring[1], pinned, LiftedVeto::None),
        "placed on the claim"
    );
}

/// The plain form follows a claim made in its gap under the same cap.
/// The followed worker is among the survivors the standing veto kept.
/// At the cap the ring places the request again. A survivor under the
/// cap takes the key. One at the cap takes the request and no pin
/// moves, though a worker over its thresholds is under the cap: this
/// form lifts no veto.
#[test]
fn a_followed_claim_at_the_cap_hands_the_key_on_under_the_standing_veto() {
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    let (ringed, _) = ring_with_a_gap(None);
    let pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&ringed, &keyed("k1"), &pool);
    let (claimed, last) = (ring[1] as usize, ring[3] as usize);
    let (pinned, handed) = (SessionVerdict::Pinned, SessionVerdict::Respill);
    let none = SessionVerdict::None;
    let under = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
        pool[claimed].session_in_flight = 1;
    });
    assert_eq!(under, (ring[1], pinned, LiftedVeto::None), "under the cap");
    let on = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
        pool[claimed].session_in_flight = 2;
    });
    assert_eq!(on, (ring[0], handed, LiftedVeto::None), "at the cap");
    let stays = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
        for candidate in pool.iter_mut() {
            candidate.session_in_flight = 2;
        }
    });
    assert_eq!(
        stays,
        (ring[0], none, LiftedVeto::None),
        "no survivor under it"
    );
    // The ring's last is under the cap and over its thresholds.
    let refused = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
        for candidate in pool.iter_mut() {
            candidate.session_in_flight = 2;
        }
        pool[last].health.overloaded = true;
        pool[last].session_in_flight = 0;
    });
    assert_eq!(
        refused,
        (ring[0], none, LiftedVeto::None),
        "no veto is lifted"
    );
    // This decision sees the ring's first worker down as the other
    // does: its own placement is the claimed worker.
    let same = followed(plain_ring_with_a_gap(rule), &ring, |pool| {
        pool[ring[0] as usize].health.healthy = false;
        pool[claimed].session_in_flight = 2;
    });
    assert_eq!(
        same,
        (ring[1], pinned, LiftedVeto::None),
        "placed on the claim"
    );
}

/// A key handed on from another request's claim at the rule's cap moves
/// its pin where it is decided, in either form, as a respill from a pin
/// read first does: the worker taken leads the key's pins ahead of its
/// dispatch, and the claimed worker stands behind it.
#[test]
fn a_key_handed_on_from_a_followed_claim_fronts_its_worker_where_it_is_decided() {
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    let (ringed, _) = ring_with_a_gap(None);
    let pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&ringed, &keyed("k1"), &pool);
    let (taken, claimed) = (ring[0] as usize, ring[1] as usize);
    let forms = [
        ("kept", ring_with_a_gap(Some(rule))),
        ("plain", plain_ring_with_a_gap(rule)),
    ];
    for (form, (engine, gap)) in forms {
        // The other decision sees the ring's first worker down and
        // claims the ring's second. This one sees the claimed worker
        // with the cap's two requests of the key.
        let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
        let mut theirs = pool.to_vec();
        theirs[taken].health.healthy = false;
        *gap.staged.lock().expect("the staged pool") = theirs;
        pool[claimed].session_in_flight = 2;
        let request = keyed("k1");
        let decision = decide(&engine, &request, &pool);
        assert!(
            gap.staged.lock().expect("the staged pool").is_empty(),
            "{form}: the other decision ran"
        );
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Respill, "{form}");
        assert_eq!(order(&decision)[0], ring[0], "{form}");
        assert_eq!(
            engine.sessions().pins(request.session()),
            Some([pool[taken].id, pool[claimed].id]),
            "{form}: ahead of its dispatch"
        );
    }
}

/// What a filter that turns was asked, and whether it turned.
#[derive(Debug, Default)]
struct Turn {
    asked: AtomicUsize,
    turned: AtomicBool,
}

/// A filter whose answer turns within a decision. It keeps every row
/// until it is asked under the lifted veto, and from then on it refuses
/// every row it is asked about under the veto as it stands.
#[derive(Debug)]
struct Turns(Arc<Turn>);

impl Filter for Turns {
    fn name(&self) -> &'static str {
        "turns"
    }
    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, cx: &Constraints<'_>) -> bool {
        self.0.asked.fetch_add(1, Ordering::Relaxed);
        if cx.overload_lifted {
            self.0.turned.store(true, Ordering::Relaxed);
        }
        cx.overload_lifted || !self.0.turned.load(Ordering::Relaxed)
    }
}

/// The hand-on's pass under the veto as it stands is put to no rule. It
/// runs where the same pass kept a worker earlier in the decision, so
/// it ends empty only under a filter whose answer turned since. The
/// lifted pass's survivors then decide and the trace says the lift, and
/// the emptied pass is not run again with the veto lifted: the filter
/// is asked about the rows of the decision's four passes and no more.
#[test]
fn a_hand_ons_pass_a_filter_emptied_is_put_to_no_rule() {
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    };
    let turn = Arc::new(Turn::default());
    let engine = ring_behind(Box::new(Turns(Arc::clone(&turn))), Some(rule));
    let (ringed, _) = ring_with_a_gap(None);
    let mut pool = [free(0, 4), free(1, 4), free(2, 4), free(3, 4)];
    let ring = ring_order(&ringed, &keyed("k1"), &pool);
    let owner = ring[0] as usize;
    assert_eq!(dispatched(&engine, &pool, 1_000), ring[0], "the pin");
    // The pin carries the cap's two requests of the key, over its
    // thresholds: the lifted pass lands on it and puts the request to
    // the ring again.
    pool[owner].session_in_flight = 2;
    pool[owner].health.overloaded = true;
    turn.turned.store(false, Ordering::Relaxed);
    turn.asked.store(0, Ordering::Relaxed);
    let request = RequestFacts {
        now_micros: 2_000,
        ..keyed("k1")
    };
    let decision = decide(&engine, &request, &pool);
    // The lifted pass's survivors hold the pin, and the ring places the
    // key on it again.
    assert_eq!(order(&decision)[0], ring[0]);
    let pick = decision.pick().expect("a pick");
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
    assert_eq!(decision.trace().lifted, LiftedVeto::Overload);
    // Two passes under the veto as it stands reach the filter with the
    // three workers under their thresholds, two lifted passes with all.
    assert_eq!(turn.asked.load(Ordering::Relaxed), 2 * 3 + 2 * 4);
}

/// A ready worker under its thresholds that the liveness veto doubts.
fn doubted(position: u32, in_flight: u32) -> Candidate {
    let mut worker = free(position, in_flight);
    worker.health.stalled = true;
    worker
}

/// A ready worker over its thresholds that the liveness veto doubts.
fn doubted_over(position: u32, in_flight: u32) -> Candidate {
    let mut worker = over(position, in_flight);
    worker.health.stalled = true;
    worker
}
