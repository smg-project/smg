//! The cases of a shadow profile's decision after a single decision in a
//! gateway's own process: each made through the engine's own entry, over a
//! list of worker objects. A shadow's answer is counted and never served:
//! the answer a request is given is the live profile's whole decision.
//! Where that decision draws by the request's seed it is held to the one
//! the same document gives with no shadow named. A rotation's whole
//! decision takes its list's own turn, the one a pick alone takes.
//!
//! The classic router asks the one policy of a request's model
//! (`placement.rs:502`) and runs no second one beside it. So a shadow's
//! count is the engine's own answer in every case here, and what the
//! request is served is not changed by it.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Instant,
};

use parking_lot::Mutex;
use smg_worker::{
    identity::basic::Worker,
    load::state::{one_rank_load, LoadSnapshot},
};

use super::super::single::{
    self,
    tests::{answered, decided, pool, Asks, MODEL},
};
use crate::{
    document::spec::{
        AffinitySource, AppliedSpec, Assignment, BestParams, PickStage, PolicySpec,
        PrefixAffinityParams, ProfileSpec, ScoreStage, ShadowSpec, WarmupSliceParams,
    },
    engine::{
        decision::{Decision, ShadowKind, ShedReason, Targets},
        decision_engine::{Decided, DecisionEngine, Request},
        demands::Demands,
        framework::{Picker, ShadowRows, TieBand},
        list_cell::tests::PublishedList,
        policy_engine::PolicyEngine,
        prompt_hashes::HashSeeds,
        records::worker_id,
        shadow::{Shadowed, SHADOW_FLOOR},
        sticky_sessions::SessionVerdict,
        view::{Candidate, Identity, Phase, RequestFacts, WorkerId},
    },
    stages::{lifecycle::Lifecycle, pickers::target_of, registry::StageRegistry},
    store::GatewayId,
    tracking::{
        lease::{EventKind, EventSet, Lease, Outcome, Side, TransportClass},
        DecisionGuard,
    },
};

/// A first attempt of [`MODEL`] that states nothing but its seed.
fn request<'a>(seed: u64) -> Request<'a> {
    let mut request = single::tests::request();
    request.facts.seed = seed;
    request
}

/// The whole decision of `request` over a value `engine` makes of `list`.
fn placed(engine: &PolicyEngine, list: &Arc<[Arc<dyn Worker>]>, request: Request<'_>) -> Decision {
    let members = PublishedList::new(Arc::clone(list));
    decided(engine, &Asks::of(request, Some(&members))).decision
}

/// The first target of a decision's pick, by its place in the list.
fn first(decision: &Decision) -> u32 {
    let pick = decision.pick().expect("a pick");
    pick.targets.first().expect("a target").position
}

/// The engine's counts of the first binding's shadow decisions: agreed,
/// another target, another outcome, skipped.
fn kinds(engine: &PolicyEngine) -> [u64; 4] {
    kinds_at(engine, 0)
}

/// [`kinds`] of the binding at `binding`.
fn kinds_at(engine: &PolicyEngine, binding: u32) -> [u64; 4] {
    ShadowKind::ALL.map(|kind| engine.shadowed(binding, kind))
}

/// The engine of `document`, under the first generation.
fn engine_of(mut document: PolicySpec, stages: &StageRegistry) -> PolicyEngine {
    document.generation = 1;
    let spec = AppliedSpec::validate_in(&document, stages, &[]);
    PolicyEngine::new(spec.expect("the document validates"))
}

/// `preset`'s document with the profile `twin` named as its binding's
/// shadow. The name sorts after every preset's own, so the live profile
/// keeps its position in the document, and a decision's trace names it as
/// it does with no shadow.
fn with_shadow_profile(preset: &str, twin: ProfileSpec) -> PolicySpec {
    let mut document = PolicySpec::preset(preset).expect("a built-in preset");
    document.profiles.insert("twin".into(), twin);
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    document
}

/// The one profile of `preset`'s document.
fn profile_of(preset: &str) -> ProfileSpec {
    let document = PolicySpec::preset(preset).expect("a built-in preset");
    let profile = document.profiles.into_values().next();
    profile.expect("the preset's profile")
}

/// What a case's picker does with a decision.
#[derive(Clone, Copy, Debug)]
enum Rule {
    /// Takes the row at this place among the rows kept, and the row behind
    /// it as its alternate.
    Place(usize),
    /// Takes nobody: its profile declines.
    Nobody,
    /// Panics.
    Panics,
    /// Takes the first row kept and the row behind it, and panics where
    /// it hears the issue of a lease.
    IssuePanics,
}

/// What the stages of a case heard of the leases issued: the stage, and
/// the worker its lease books, by its id.
type Heard = Arc<Mutex<Vec<(&'static str, WorkerId)>>>;

/// A picker under a case's rule. It counts the decisions it picks for and
/// those that hand it affinity rows, and keeps the lease issues it hears.
#[derive(Debug)]
struct Takes {
    name: &'static str,
    rule: Rule,
    asked: Arc<AtomicU64>,
    named: Arc<AtomicU64>,
    heard: Heard,
}

impl Lifecycle for Takes {
    fn hears(&self) -> EventSet {
        EventSet::NONE.with(EventKind::Issued)
    }

    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        assert!(
            !matches!(self.rule, Rule::IssuePanics),
            "the case's own panic, at a lease's issue"
        );
        self.heard.lock().push((self.name, lease.booked));
    }
}

impl Picker for Takes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        self.asked.fetch_add(1, Ordering::Relaxed);
        let named = u64::from(facts.named.is_some());
        self.named.fetch_add(named, Ordering::Relaxed);
        let place = match self.rule {
            Rule::Place(place) => place,
            Rule::IssuePanics => 0,
            Rule::Nobody => return,
            Rule::Panics => panic!("the case's own panic, inside a shadow"),
        };
        for &row in kept.iter().skip(place).take(2) {
            out.push(target_of(&candidates[row as usize]));
        }
    }
}

/// What a case reads of its two pickers: the live profile's and the shadow
/// profile's.
#[derive(Default)]
struct Seen {
    /// The decisions that asked each picker.
    asked: [Arc<AtomicU64>; 2],
    /// Those of them that handed it named rows.
    named: [Arc<AtomicU64>; 2],
    heard: Heard,
}

impl Seen {
    /// The stages that hold the two pickers, under the names `takes-live`
    /// and `takes-shadow`.
    fn stages(&self, live: Rule, shadow: Rule) -> StageRegistry {
        let mut stages = StageRegistry::builtin();
        for (at, name, rule) in [(0, "takes-live", live), (1, "takes-shadow", shadow)] {
            let (asked, named) = (Arc::clone(&self.asked[at]), Arc::clone(&self.named[at]));
            let heard = Arc::clone(&self.heard);
            let added = stages.picker(name, move || -> Box<dyn Picker> {
                Box::new(Takes {
                    name,
                    rule,
                    asked: Arc::clone(&asked),
                    named: Arc::clone(&named),
                    heard: Arc::clone(&heard),
                })
            });
            added.expect("a free name");
        }
        stages
    }

    /// How many decisions asked the shadow profile's picker.
    fn shadows(&self) -> u64 {
        self.asked[1].load(Ordering::Relaxed)
    }
}

/// The engine of `preset`'s document with the picker `live` in its pick's
/// place and, where `shadow` names a rule, a second profile of the same
/// stages under that picker as its binding's shadow. `edit` has the
/// document before it is validated.
fn bed_of(
    preset: &str,
    live: Rule,
    shadow: Option<Rule>,
    edit: impl FnOnce(&mut PolicySpec),
) -> (PolicyEngine, Seen) {
    let seen = Seen::default();
    let stages = seen.stages(live, shadow.unwrap_or(Rule::Nobody));
    let mut profile = profile_of(preset);
    profile.pick = "takes-live".into();
    let mut twin = profile.clone();
    twin.pick = "takes-shadow".into();
    let mut document = match shadow {
        Some(_) => with_shadow_profile(preset, twin),
        None => PolicySpec::preset(preset).expect("a built-in preset"),
    };
    let own = document.profiles.get_mut(preset);
    *own.expect("the preset's profile") = profile;
    edit(&mut document);
    (engine_of(document, &stages), seen)
}

/// [`bed_of`] under the `random` preset's stages, with a shadow.
fn bed(live: Rule, shadow: Rule) -> (PolicyEngine, Seen) {
    bed_of("random", live, Some(shadow), |_| {})
}

/// A shadow that picks another worker is counted as one, and the answer is
/// the live profile's: the decision given beside a shadow that agrees, and
/// the one the document gives with no shadow. One lease is issued, on the
/// live pick's worker, and the stages of both profiles hear that issue and
/// no other.
#[test]
fn a_shadow_that_picks_another_worker_is_counted_and_the_live_answer_served() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::Place(0), Rule::Place(1));
    let members = PublishedList::new(Arc::clone(&list));
    let answer = decided(&engine, &Asks::of(request(7), Some(&members)));
    assert_eq!(kinds(&engine), [0, 1, 0, 0]);
    assert_eq!(seen.shadows(), 1);
    let pick = answer.decision.pick().expect("a pick");
    let live = pick.targets.first().expect("a target").id;
    let lease = answer.lease.expect("a lease").lease;
    assert_eq!((lease.booked, lease.id.0), (live, 1));
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 1, "one id taken");
    let heard = [("takes-live", live), ("takes-shadow", live)];
    assert_eq!(*seen.heard.lock(), heard);

    let (agreeing, _) = bed(Rule::Place(0), Rule::Place(0));
    assert_eq!(placed(&agreeing, &list, request(7)), answer.decision);
    assert_eq!(kinds(&agreeing), [1, 0, 0, 0]);
    let (unshadowed, seen) = bed_of("random", Rule::Place(0), None, |_| {});
    assert_eq!(placed(&unshadowed, &list, request(7)), answer.decision);
    assert_eq!(kinds(&unshadowed), [0; 4]);
    assert_eq!(seen.shadows(), 0);
}

/// A shadow that sheds leaves the live pick as it was decided. A live
/// decision that sheds is answered as its shed, with no lease, whatever its
/// shadow picks. Each is counted as another outcome.
#[test]
fn a_shadow_of_another_outcome_changes_nothing_of_the_live_answer() {
    let list = pool(3);
    let (unshadowed, _) = bed_of("random", Rule::Place(0), None, |_| {});
    let (engine, _) = bed(Rule::Place(0), Rule::Nobody);
    let answer = placed(&engine, &list, request(7));
    assert_eq!(answer, placed(&unshadowed, &list, request(7)));
    assert_eq!(kinds(&engine), [0, 0, 1, 0]);

    let (engine, seen) = bed(Rule::Nobody, Rule::Place(0));
    let members = PublishedList::new(Arc::clone(&list));
    let answer = decided(&engine, &Asks::of(request(7), Some(&members)));
    assert!(matches!(answer.decision, Decision::Shed(_)), "{answer:?}");
    assert!(answer.lease.is_none());
    assert!(seen.heard.lock().is_empty(), "no lease, no issue");
    assert_eq!(seen.shadows(), 1);
    assert_eq!(kinds(&engine), [0, 0, 1, 0]);
}

/// A request that states what no stage here runs is refused under that
/// stage's name, as it is where no shadow is named. No profile decided it:
/// the shadow's picker does not run, and no shadow decision is counted.
/// A request whose demand no worker of the list meets is shed ahead of
/// the decision alike, under the demand's stage.
#[test]
fn a_request_refused_ahead_of_the_decision_has_no_shadow() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::Place(0), Rule::Place(1));
    let shed = |stated| match placed(&engine, &list, stated) {
        Decision::Shed(shed) => Ok((shed.reason, shed.emptied_by)),
        decision => Err(decision.pick().is_some()),
    };
    let mut hinted = request(7);
    hinted.constraints.subset_hint = Some(2);
    assert_eq!(
        shed(hinted),
        Ok((ShedReason::PolicyDeclined, "subset-hint"))
    );
    // The list's host said nothing of its workers: none takes images.
    let mut demanding = request(7);
    demanding.constraints.demands = Demands {
        vision: true,
        ..Demands::default()
    };
    assert_eq!(shed(demanding), Ok((ShedReason::NoEndpoints, "vision")));
    assert_eq!(seen.shadows(), 0);
    assert_eq!(kinds(&engine), [0; 4]);
}

/// A request with less than the floor of its stated budget keeps it for its
/// answer: its shadow profile does not run, and is counted as skipped. At
/// the floor, and with no budget stated, the shadow profile runs. The
/// answer is the same each time.
#[test]
fn a_budget_under_the_floor_skips_the_shadow_and_keeps_the_answer() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::Place(0), Rule::Place(1));
    let floor = u32::try_from(SHADOW_FLOOR.as_micros()).expect("a floor in microseconds");
    let budgeted = |micros: u32| {
        let mut stated = request(7);
        stated.facts.budget_micros = micros;
        stated
    };
    let short = placed(&engine, &list, budgeted(floor - 1));
    assert_eq!(seen.shadows(), 0);
    assert_eq!(kinds(&engine), [0, 0, 0, 1]);
    for stated in [budgeted(floor), request(7)] {
        assert_eq!(first(&placed(&engine, &list, stated)), first(&short));
    }
    assert_eq!(seen.shadows(), 2);
    assert_eq!(kinds(&engine), [0, 2, 0, 1]);
}

/// A shadow that panics costs its request nothing: the answer and its lease
/// are the live decision's, the fault is counted with the engine's, no
/// shadow decision is counted, and the next request is decided.
#[test]
fn a_shadow_that_panics_leaves_the_live_answer_and_counts_a_fault() {
    let list = pool(3);
    let (unshadowed, _) = bed_of("random", Rule::Place(0), None, |_| {});
    let (engine, seen) = bed(Rule::Place(0), Rule::Panics);
    let members = PublishedList::new(Arc::clone(&list));
    for faults in 1..=2 {
        let answer = decided(&engine, &Asks::of(request(7), Some(&members)));
        assert_eq!(answer.decision, placed(&unshadowed, &list, request(7)));
        let lease = answer.lease.expect("the live pick's lease").lease;
        assert_eq!(lease.id.0, faults, "each answer takes the next id");
        assert_eq!(engine.faults(), faults);
        assert_eq!(kinds(&engine), [0; 4]);
    }
    assert_eq!(seen.heard.lock().len(), 4, "each issue, at both stages");
}

/// A shadow decision is counted behind the issue of its live decision's
/// lease. A stage that panics at that issue is held and counted with the
/// words a stage did not hear: the request keeps its decision and its
/// lease, no fault is counted, and the shadow that ran beside it is counted
/// as beside any decision. Before, the request had no decision and its
/// shadow counted nothing.
#[test]
fn a_decision_whose_issue_panics_counts_its_shadow() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::IssuePanics, Rule::Place(1));
    let members = PublishedList::new(Arc::clone(&list));
    let answer = decided(&engine, &Asks::of(request(7), Some(&members)));
    let pick = answer.decision.pick().expect("a pick");
    let live = pick.targets.first().expect("a target").id;
    let lease = answer.lease.expect("the live pick's lease").lease;
    assert_eq!((lease.booked, lease.id.0), (live, 1));
    assert_eq!(seen.shadows(), 1, "the shadow profile ran");
    assert_eq!((engine.faults(), engine.unheard()), (0, 1));
    // The shadow's picker takes the row behind the live pick's.
    assert_eq!(kinds(&engine), [0, 1, 0, 0]);
    // The live profile's stage panicked: the stage behind it heard nothing.
    assert!(seen.heard.lock().is_empty());
}

/// The entry that issues no lease runs a binding's shadow behind the live
/// decision, as the leasing entry does, and answers what the shadow said
/// beside the live profile's whole decision: its binding, how it compares
/// and its first target. No id is taken and no stage hears an issue. The
/// leasing entry of an engine made alike gives the same decision.
#[test]
fn a_decision_without_a_lease_answers_what_its_shadow_said() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::Place(0), Rule::Place(1));
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(7), Some(&members));
    let answer = engine.decide_without_lease(&asks, Instant::now());
    let (alone, said) = answer.expect("a decision");
    assert_eq!(seen.shadows(), 1, "the shadow profile ran");
    // The shadow's picker takes the row behind the live pick's: the live
    // pick's alternate.
    let pick = alone.pick().expect("a pick");
    let alternate = pick.targets.as_slice().get(1).expect("an alternate").id;
    let said_of = Shadowed {
        binding: 0,
        kind: ShadowKind::OtherTarget,
        first: Some(alternate),
    };
    assert_eq!(said, Some(said_of));
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
    assert!(seen.heard.lock().is_empty(), "no lease, no issue");

    let (leasing, _) = bed(Rule::Place(0), Rule::Place(1));
    assert_eq!(placed(&leasing, &list, request(7)), alone);
}

/// What a shadow said beside a decision with no lease is counted when the
/// entry's caller says so, and not before: the caller counts behind its own
/// filing of the lease, as the leasing entry counts behind its issue. One
/// call counts one shadow decision, and a caller whose filing failed calls
/// nothing. A decision that had no shadow hands none back, and counting
/// that allocates no table.
#[test]
fn a_shadow_beside_a_decision_without_a_lease_is_counted_by_its_caller() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::Place(0), Rule::Place(1));
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(7), Some(&members));
    let answer = engine.decide_without_lease(&asks, Instant::now());
    let (_, said) = answer.expect("a decision");
    assert!(said.is_some(), "the shadow's word");
    assert!(
        engine.document.shadow_counts.get().is_none(),
        "nothing counted yet"
    );
    assert_eq!(kinds(&engine), [0; 4]);
    engine.count_shadow(said);
    assert_eq!(kinds(&engine), [0, 1, 0, 0]);

    // A filing that failed: the word is dropped, and nothing more counts.
    let answer = engine.decide_without_lease(&asks, Instant::now());
    let (_, dropped) = answer.expect("a decision");
    assert!(dropped.is_some(), "the shadow's word");
    assert_eq!(seen.shadows(), 2);
    assert_eq!(kinds(&engine), [0, 1, 0, 0]);

    let (unshadowed, seen) = bed_of("random", Rule::Place(0), None, |_| {});
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(7), Some(&members));
    let answer = unshadowed.decide_without_lease(&asks, Instant::now());
    let (_, none) = answer.expect("a decision");
    assert_eq!(none, None);
    unshadowed.count_shadow(none);
    assert_eq!(seen.shadows(), 0);
    assert!(unshadowed.document.shadow_counts.get().is_none());
}

/// A live decision that sheds files no lease, and the entry still answers
/// what its shadow said beside the shed: another outcome, with the shadow's
/// own first target. Its caller counts that all the same, as the leasing
/// entry counts a live shed's shadow.
#[test]
fn a_shed_without_a_lease_answers_its_shadows_word_for_the_count() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::Nobody, Rule::Place(0));
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(7), Some(&members));
    let answer = engine.decide_without_lease(&asks, Instant::now());
    let (shed, said) = answer.expect("a decision");
    assert!(matches!(shed, Decision::Shed(_)), "{shed:?}");
    assert_eq!(seen.shadows(), 1);
    let said = said.expect("the shadow's word");
    assert_eq!((said.binding, said.kind), (0, ShadowKind::OtherOutcome));
    assert!(said.first.is_some(), "the shadow's own pick");
    assert_eq!(kinds(&engine), [0; 4]);
    engine.count_shadow(Some(said));
    assert_eq!(kinds(&engine), [0, 0, 1, 0]);

    let (leasing, _) = bed(Rule::Nobody, Rule::Place(0));
    assert_eq!(placed(&leasing, &list, request(7)), shed);
    assert_eq!(kinds(&leasing), [0, 0, 1, 0]);
}

/// A shadow profile decides under its request's own constraints: a worker
/// the host just failed to reach is no pick of the shadow's either. So a
/// shadow that takes the place the live profile takes agrees with it.
#[test]
fn a_shadow_decides_under_its_requests_constraints() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::Place(0), Rule::Place(0));
    let members = PublishedList::new(Arc::clone(&list));
    let unreached = [worker_id(list[0].as_ref())];
    let mut asked = request(7);
    asked.constraints.excluded = &unreached;
    let answer = decided(&engine, &Asks::of(asked, Some(&members)));
    assert_eq!(first(&answer.decision), 1, "the worker behind the first");
    assert_eq!(seen.shadows(), 1);
    assert_eq!(kinds(&engine), [1, 0, 0, 0], "the shadow agrees");
}

/// What a shadow said is handed to the caller that counts it: its
/// binding's position, how it compares with the live decision, and the
/// first target of its own pick. Nothing is counted ahead of that caller's
/// issue.
#[test]
fn what_a_shadow_said_names_its_binding_its_kind_and_its_first_target() {
    let list = pool(3);
    let (engine, _) = bed(Rule::Place(0), Rule::Place(1));
    let members = PublishedList::new(Arc::clone(&list));
    let word = engine.spec().engine().shadow_for(MODEL, Phase::Single);
    let word = word.expect("the binding's shadow");
    let mut asked = request(7);
    // The request's maker, which names no entry in a token tree.
    let maker = Asks::of(asked, Some(&members));
    let now = Instant::now();
    let made = engine.members_of(members.lent());
    let (decision, said) =
        super::decide(engine.loaded(), &mut asked, made, &maker, None, now, word);
    // The shadow's picker takes the row behind the live pick's: the live
    // pick's alternate.
    let pick = decision.pick().expect("a pick");
    let alternate = pick.targets.as_slice().get(1).expect("an alternate").id;
    let said_of = Shadowed {
        binding: 0,
        kind: ShadowKind::OtherTarget,
        first: Some(alternate),
    };
    assert_eq!(said, Some(said_of));
    assert_eq!(kinds(&engine), [0; 4]);
}

/// A shadow profile equal to the live one agrees with it on every decision
/// of a run: both read the records written once for the two of them. The
/// answers are the ones the document gives with no shadow, with and without
/// load reports. Over a list this size a profile that draws writes, with no
/// shadow, the records of the workers it draws alone. With a shadow every
/// worker's record is written, and the draw is made over them: the same
/// workers are picked, in the same order.
#[test]
fn a_shadow_equal_to_the_live_profile_agrees_at_every_seed() {
    use crate::engine::framework::DRAW_FROM;

    const WORKERS: usize = 2 * DRAW_FROM;
    let list = pool(WORKERS);
    // Queues of every length, no two workers in a row alike.
    let reports = (0..WORKERS).map(|at| {
        let queued = i32::try_from(at * 7 % 13).expect("a small count") * 4_000;
        let url = format!("http://worker-{at}:8000");
        (url, one_rank_load(queued, 0.1, 1_000.0))
    });
    let snapshot = LoadSnapshot::from_loads_for_test(reports.collect());
    let stages = StageRegistry::builtin();
    let answer = |engine: &PolicyEngine, loads: &Option<Arc<LoadSnapshot>>, seed: u64| {
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks {
            loads: loads.clone(),
            ..Asks::of(request(seed), Some(&members))
        };
        // A plain `random` request with no shadow is answered a place: its
        // whole decision is the one the entry that issues no lease makes.
        match answered(engine, &asks).whole() {
            Some((decided, _)) => decided.decision,
            None => {
                let whole = engine.decide_without_lease(&asks, Instant::now());
                whole.expect("a whole decision").0
            }
        }
    };
    for preset in ["least_load", "power_of_two", "random"] {
        let engine = engine_of(with_shadow_profile(preset, profile_of(preset)), &stages);
        let unshadowed = engine_of(
            PolicySpec::preset(preset).expect("a built-in preset"),
            &stages,
        );
        for loads in [None, Some(Arc::clone(&snapshot))] {
            for seed in 0..64 {
                assert_eq!(
                    answer(&engine, &loads, seed),
                    answer(&unshadowed, &loads, seed),
                    "{preset}, seed {seed}, reports {}",
                    loads.is_some()
                );
            }
        }
        assert_eq!(kinds(&engine), [128, 0, 0, 0], "{preset}");
    }
}

/// Round robin's plain request is answered by one pick of its picker, with
/// no request made. Under a binding that names a shadow it is made and
/// takes the whole decision, where its shadow profile runs: the live
/// profile still takes the workers in turn. A binding of the same document
/// that names no shadow keeps its one-pick answer, and its made requests
/// count no shadow decision.
#[test]
fn a_plain_request_of_a_binding_with_a_shadow_takes_the_whole_decision() {
    let stages = StageRegistry::builtin();
    let list = pool(4);
    let mut document = with_shadow_profile("round_robin", profile_of("round_robin"));
    let mut named = document.bindings[0].clone();
    named.matcher.model = "alpha-*".to_owned();
    named.shadow = None;
    document.bindings.insert(0, named);
    let engine = engine_of(document, &stages);
    let members = PublishedList::new(Arc::clone(&list));

    let asks = Asks::of(request(7), Some(&members));
    let mut seen = [0_u32; 4];
    for _ in 0..8 {
        seen[first(&decided(&engine, &asks).decision) as usize] += 1;
    }
    assert_eq!(seen, [2; 4], "every worker takes its turn");
    assert_eq!(asks.made.get(), 8, "each request is made");
    assert_eq!(kinds_at(&engine, 1).iter().sum::<u64>(), 8);
    assert_eq!(kinds_at(&engine, 1)[3], 0, "each shadow profile ran");

    // The first binding names no shadow: a list of its model is another
    // value, as a gateway's list is one model's.
    let other = PublishedList::new(Arc::clone(&list));
    let mut unshadowed = request(7);
    unshadowed.facts.model = "alpha-1";
    let asks = Asks::of(unshadowed, Some(&other));
    assert!(matches!(answered(&engine, &asks), Decided::Alone(_)));
    assert_eq!(asks.made.get(), 0, "the pick read no fact of a request");
    let tried = [WorkerId(1)];
    let mut retry = unshadowed;
    retry.facts.tried = &tried;
    let asks = Asks::of(retry, Some(&other));
    assert!(decided(&engine, &asks).decision.pick().is_some());
    assert_eq!(kinds_at(&engine, 0), [0; 4]);
    assert_eq!(kinds_at(&engine, 1).iter().sum::<u64>(), 8);

    // A list value keeps the word of its first request's model. A request
    // of the other model over it is answered by the document at its call:
    // no shadow for the first binding's, one for the second binding's.
    let asks = Asks::of(retry, Some(&members));
    assert!(decided(&engine, &asks).decision.pick().is_some());
    assert_eq!(kinds_at(&engine, 1).iter().sum::<u64>(), 8);
    let _ = decided(&engine, &Asks::of(request(7), Some(&other)));
    assert_eq!(kinds_at(&engine, 1).iter().sum::<u64>(), 9);
    assert_eq!(kinds_at(&engine, 0), [0; 4]);
}

/// Round robin under a binding that names a shadow takes its list's workers
/// one after the other, as its one pick does with no shadow named: the
/// whole decision takes the list's own turn at every size, and the shadow's
/// run, a round robin of its own, moves no turn of it. So the list's word
/// moves once a request. The classic router turns the list it is handed
/// whole (`model_gateway/src/policies/round_robin.rs:151`) and runs no
/// second policy beside it.
#[test]
fn a_shadow_leaves_the_live_rotation_one_turn_a_request() {
    let stages = StageRegistry::builtin();
    for workers in [3, 16, 64] {
        let document = with_shadow_profile("round_robin", profile_of("round_robin"));
        let engine = engine_of(document, &stages);
        let members = PublishedList::new(pool(workers));
        let word = &engine.members_of(members.lent()).rotation.0;
        let before = word.load(Ordering::Relaxed);
        let asks = Asks::of(request(7), Some(&members));
        let places: Vec<usize> = (0..2 * workers)
            .map(|_| first(&decided(&engine, &asks).decision) as usize)
            .collect();
        let in_turn: Vec<usize> = (0..2 * workers)
            .map(|turn| (places[0] + turn) % workers)
            .collect();
        assert_eq!(places, in_turn, "{workers} workers");
        let turned = word.load(Ordering::Relaxed) - before;
        assert_eq!(turned, 2 * workers as u64, "{workers} workers");
        assert_eq!(kinds(&engine)[3], 0, "each shadow profile ran");
        assert_eq!(kinds(&engine).iter().sum::<u64>(), 2 * workers as u64);
    }
}

/// The default policy's prompt goes back to the worker it was sent to, by
/// the engine's record of sent prefixes. Its shadow writes nothing there:
/// each prompt returns to its live pick at every seed, never to the worker
/// the shadow picked, and every answer is the one the document gives with
/// no shadow. The shadow profile reads no affinity column, so it runs
/// without the rows named for the live one.
#[test]
fn a_shadow_writes_nothing_to_the_record_of_sent_prefixes() {
    let seen = Seen::default();
    let stages = seen.stages(Rule::Nobody, Rule::Place(0));
    let mut twin = profile_of("random");
    twin.pick = "takes-shadow".into();
    let engine = engine_of(with_shadow_profile("cache_aware", twin), &stages);
    let unshadowed = engine_of(
        PolicySpec::preset("cache_aware").expect("a built-in preset"),
        &stages,
    );
    let word = engine.spec().engine().shadow_for(MODEL, Phase::Single);
    assert_eq!(word, Some((0, ShadowRows::Unnamed)));

    let list = pool(8);
    let (members, plain) = (
        PublishedList::new(Arc::clone(&list)),
        PublishedList::new(Arc::clone(&list)),
    );
    let prompt = |start: u32| (start..start + 1_024).collect::<Vec<u32>>();
    let sent = |tokens: &[u32], seed: u64| {
        let answer = |engine: &PolicyEngine, members: &PublishedList| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request(seed);
            hashes.write_to(&mut asked.facts);
            decided(engine, &Asks::of(asked, Some(members))).decision
        };
        let shadowed = answer(&engine, &members);
        assert_eq!(shadowed, answer(&unshadowed, &plain), "seed {seed}");
        first(&shadowed)
    };
    let prompts: Vec<Vec<u32>> = (1..=16).map(|at| prompt(at * 100_000)).collect();
    let picks: Vec<u32> = (0_u64..)
        .zip(&prompts)
        .map(|(seed, tokens)| sent(tokens, seed))
        .collect();
    // The shadow takes the first worker each time: a record that kept its
    // pick would send every prompt there.
    assert!(picks.iter().any(|&position| position != 0), "{picks:?}");
    for seed in 100..108 {
        for (tokens, &position) in prompts.iter().zip(&picks) {
            assert_eq!(sent(tokens, seed), position, "seed {seed}");
        }
    }
    let asked = 16 + 8 * 16;
    assert_eq!(seen.shadows(), asked);
    assert_eq!(kinds(&engine).iter().sum::<u64>(), asked);
    assert_eq!(kinds(&engine)[3], 0, "each shadow profile ran");
    let named = seen.named[1].load(Ordering::Relaxed);
    assert_eq!(named, 0, "the shadow profile is handed no affinity row");
    assert_eq!(engine.affinity().left_out(), 0);
}

/// A shadow profile that reads no affinity column, under a live profile
/// that is handed affinity rows, runs without them: the rows would turn a
/// sampling picker's pick to a draw. The live profile's picker is handed
/// the rows at the same decisions.
#[test]
fn a_shadow_that_reads_no_affinity_runs_without_the_live_profiles_rows() {
    let seen = Seen::default();
    let stages = seen.stages(Rule::Place(0), Rule::Place(0));
    let mut live = profile_of("cache_aware");
    live.pick = "takes-live".into();
    let mut twin = profile_of("random");
    twin.pick = "takes-shadow".into();
    let mut document = with_shadow_profile("cache_aware", twin);
    let own = document.profiles.get_mut("cache_aware");
    *own.expect("the preset's profile") = live;
    let engine = engine_of(document, &stages);
    let word = engine.spec().engine().shadow_for(MODEL, Phase::Single);
    assert_eq!(word, Some((0, ShadowRows::Unnamed)));

    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..1_024).collect();
    let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
    for seed in 0..2 {
        let mut asked = request(seed);
        hashes.write_to(&mut asked.facts);
        let _ = decided(&engine, &Asks::of(asked, Some(&members)));
    }
    let load = |count: &Arc<AtomicU64>| count.load(Ordering::Relaxed);
    assert_eq!(seen.asked.each_ref().map(load), [2, 2]);
    assert_eq!(seen.named.each_ref().map(load), [2, 0]);
    assert_eq!(kinds(&engine), [2, 0, 0, 0]);
}

/// A shadow profile that reads the live profile's own rows is handed them:
/// under one sample and one ruling the two are named the same rows, and the
/// shadow's picker sees them at each decision.
#[test]
fn a_shadow_that_reads_the_live_profiles_rows_is_handed_them() {
    let seen = Seen::default();
    let stages = seen.stages(Rule::Place(0), Rule::Place(0));
    let mut live = profile_of("cache_aware");
    live.pick = "takes-live".into();
    let mut twin = live.clone();
    twin.pick = "takes-shadow".into();
    let mut document = with_shadow_profile("cache_aware", twin);
    let own = document.profiles.get_mut("cache_aware");
    *own.expect("the preset's profile") = live;
    let engine = engine_of(document, &stages);
    let word = engine.spec().engine().shadow_for(MODEL, Phase::Single);
    assert_eq!(word, Some((0, ShadowRows::Same)));

    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..1_024).collect();
    let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
    for seed in 0..2 {
        let mut asked = request(seed);
        hashes.write_to(&mut asked.facts);
        let _ = decided(&engine, &Asks::of(asked, Some(&members)));
    }
    let load = |count: &Arc<AtomicU64>| count.load(Ordering::Relaxed);
    assert_eq!(seen.asked.each_ref().map(load), [2, 2]);
    assert_eq!(seen.named.each_ref().map(load), [2, 2]);
    assert_eq!(kinds(&engine), [2, 0, 0, 0]);
}

/// A list keeps one ruling of its warming rows and one record of sent
/// prefixes: the live profile's. A shadow profile of another ruling, or one
/// that reads the record alone where the live profile reads cache events
/// too, does not run. Nor does one that would be handed affinity rows of
/// its own: a profile of another sample, and one that reads who holds the
/// prompt under a live profile that reads no affinity. Each is counted as
/// skipped. A shadow profile of the live profile's own sample and ruling
/// runs over the same rows.
#[test]
fn a_shadow_with_other_affinity_rows_than_the_live_profile_is_skipped() {
    let stages = StageRegistry::builtin();
    let shadowed = |live: &str, turn: &dyn Fn(&mut ProfileSpec)| {
        let mut twin = profile_of("cache_aware");
        turn(&mut twin);
        let engine = engine_of(with_shadow_profile(live, twin), &stages);
        let word = engine.spec().engine().shadow_for(MODEL, Phase::Single);
        let _ = placed(&engine, &pool(3), request(7));
        (word.map(|(_, rows)| rows), kinds(&engine))
    };
    let sample = |sample: u32| {
        move |twin: &mut ProfileSpec| twin.pick = PickStage::Best(BestParams { sample }).into()
    };
    let ruling = |twin: &mut ProfileSpec| {
        let slice = WarmupSliceParams {
            weight: 8.0,
            secs: Some(120),
            ..WarmupSliceParams::default()
        };
        twin.score[0] = ScoreStage::WarmupSlice(slice).into();
    };
    let record = |twin: &mut ProfileSpec| {
        let prefix = PrefixAffinityParams {
            weight: 4.0,
            source: Some(AffinitySource::Predictor),
            ..PrefixAffinityParams::default()
        };
        twin.score[1] = ScoreStage::PrefixAffinity(prefix).into();
    };
    let apart = (Some(ShadowRows::Apart), [0, 0, 0, 1]);
    assert_eq!(shadowed("cache_aware", &ruling), apart);
    assert_eq!(shadowed("cache_aware", &record), apart);
    let own = (Some(ShadowRows::Own), [0, 0, 0, 1]);
    assert_eq!(shadowed("cache_aware", &sample(8)), own);
    assert_eq!(shadowed("random", &|_| {}), own);
    let (word, kinds) = shadowed("cache_aware", &sample(16));
    assert_eq!(word, Some(ShadowRows::Same));
    assert_eq!((kinds.iter().sum::<u64>(), kinds[3]), (1, 0), "{kinds:?}");
}

/// A fresh key under a sessions rule is the shadow profile's to place too.
/// It reads the sessions as the live decision found them, with no pin, and
/// picks its own worker. The key is pinned where the live answer was sent
/// and nowhere else: its next request stays, and the shadow follows that
/// pin too. The shadow claims no pin of its own.
#[test]
fn a_shadow_under_a_sessions_rule_moves_no_session_pin() {
    let (engine, _) = bed_of("manual", Rule::Place(0), Some(Rule::Place(1)), |document| {
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").assignment = Assignment::Delegate;
    });
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let mut keyed = request(7);
    keyed.facts.session_key = Identity::hashed(7);
    let asks = Asks::of(keyed, Some(&members));
    let now = Instant::now();
    let mut held =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let placed = *held.decision().pick().expect("a pick");
    let owner = placed.targets.first().expect("a target").id;
    assert_eq!(placed.session_verdict, SessionVerdict::Claimed(owner));
    assert_eq!(kinds(&engine), [0, 1, 0, 0]);
    let taken = held.take(0).expect("the first target");
    held.dispatched(now);
    held.completed(
        &Outcome {
            side: Side::Single,
            served: taken.id,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        },
        now,
    );
    drop(held);

    let next = decided(&engine, &asks).decision;
    let pick = next.pick().expect("a pick");
    let stayed = pick.targets.first().expect("a target").id;
    assert_eq!(
        (stayed, pick.session_verdict),
        (owner, SessionVerdict::Pinned)
    );
    assert_eq!(kinds(&engine), [1, 1, 0, 0]);
    assert_eq!(engine.spec().engine().sessions().len(), 1);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// Every thread counts its shadow decisions in the engine's one table:
/// four threads' counts are all read by a thread that counted none. A
/// document that names no shadow allocates no table.
#[test]
fn the_shadow_counts_of_every_thread_are_read_as_one() {
    let list = pool(3);
    let (engine, seen) = bed(Rule::Place(0), Rule::Place(1));
    assert!(
        engine.document.shadow_counts.get().is_none(),
        "no count yet"
    );
    thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                let members = PublishedList::new(Arc::clone(&list));
                let asks = Asks::of(request(7), Some(&members));
                for _ in 0..25 {
                    let _ = decided(&engine, &asks);
                }
            });
        }
    });
    assert_eq!(kinds(&engine), [0, 100, 0, 0]);
    assert_eq!(seen.shadows(), 100);

    let (unshadowed, _) = bed_of("random", Rule::Place(0), None, |_| {});
    let _ = placed(&unshadowed, &list, request(7));
    assert!(unshadowed.document.shadow_counts.get().is_none());
}
