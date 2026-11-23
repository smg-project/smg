//! The cases of a single request's words under a sessions rule: what a
//! stage hears of a pick by who placed it and by the rule's word of what it
//! places, asked through the engine's own entry over a list of worker
//! objects.

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Instant,
};

use smg_worker::{identity::basic::Worker, test_support::ready_worker};

use crate::{
    document::spec::{AppliedSpec, Assignment, PolicySpec, RulePlaced},
    engine::{
        decision::{Decision, PlacedBy, Target, Targets},
        decision_engine::{Request, RequestMaker, WorkerLists},
        framework::{Picker, TieBand},
        list_cell::tests::PublishedList,
        policy_engine::PolicyEngine,
        view::{Candidate, Constraints, Identity, RequestFacts},
    },
    policies::{power_of_two::PowerOfTwoMetric, random::Random},
    stages::{registry::StageRegistry, Lifecycle},
    store::GatewayId,
    tracking::{
        lease::{EventSet, Lease, Outcome, ReleaseReason, Side, TransportClass},
        DecisionGuard,
    },
};

const MODEL: &str = "model-a";
/// Every place of a list of three workers.
const EVERY: [u32; 3] = [0, 1, 2];

/// The issues and the dispatches a stage heard.
#[derive(Debug, Default)]
struct Heard {
    issued: AtomicU32,
    dispatched: AtomicU32,
}

impl Heard {
    fn words(&self) -> (u32, u32) {
        let read = |word: &AtomicU32| word.load(Ordering::Relaxed);
        (read(&self.issued), read(&self.dispatched))
    }
}

/// The `random` picker, counting the issues and the dispatches it hears.
#[derive(Debug)]
struct Hears(Arc<Heard>);

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        self.0.issued.fetch_add(1, Ordering::Relaxed);
    }

    fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
        self.0.dispatched.fetch_add(1, Ordering::Relaxed);
    }
}

impl Picker for Hears {
    fn name(&self) -> &'static str {
        "hears"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The `manual` preset's document with the profiles of `stock` in place of
/// its own, bound under the preset's sessions rule: `assignment`, a cap of
/// two requests of a key on a worker, and `kept`, the rule's word of a
/// request it places.
fn ruled(stock: PolicySpec, assignment: Assignment, kept: RulePlaced) -> PolicySpec {
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    let name = stock.profiles.keys().next().expect("a profile").clone();
    document.profiles = stock.profiles;
    let binding = &mut document.bindings[0];
    binding.profile = Some(name);
    let rule = binding.sessions.as_mut().expect("the stock rule");
    rule.assignment = assignment;
    rule.inflight_cap = 2;
    rule.rule_placed = kept;
    document.generation = 1;
    document
}

/// The engine of [`ruled`] over the `random` profile, picking with
/// [`Hears`].
fn hearing(assignment: Assignment, kept: RulePlaced, heard: &Arc<Heard>) -> PolicyEngine {
    let mut stages = StageRegistry::builtin();
    let counted = Arc::clone(heard);
    let added = stages.picker("hears", move || Box::new(Hears(Arc::clone(&counted))));
    added.expect("a free name");
    let random = PolicySpec::preset("random").expect("a built-in preset");
    let mut document = ruled(random, assignment, kept);
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "hears".into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    PolicyEngine::new(spec.expect("the stages hold the picker"))
}

/// A request stated ahead, with the list its host hands for it.
struct Asks<'a> {
    request: Request<'a>,
    members: &'a PublishedList,
}

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members.lent())
    }
}

/// The requests a case holds in flight.
type Held<'e> = Vec<DecisionGuard<&'e PolicyEngine>>;

/// One request of `key`, with no key where it is empty, sent to the target
/// at `take` of its pick as its host sends it: the dispatch is said, and
/// the worker object counts the request and its key. Where `places` says
/// so its host holds the request to every place of its list, which makes
/// it the whole decision's to answer. The request stays in flight in
/// `held`. The place it went to, and who placed it.
fn sent<'e>(
    engine: &'e PolicyEngine,
    list: &[Arc<dyn Worker>],
    members: &PublishedList,
    (key, take, places): (&'static str, usize, bool),
    held: &mut Held<'e>,
) -> (usize, PlacedBy) {
    let session_key = match key {
        "" => Identity::NONE,
        key => Identity::plain(key),
    };
    let constraints = Constraints {
        places: places.then_some(&EVERY[..]),
        ..Constraints::default()
    };
    let asks = Asks {
        request: Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                seed: held.len() as u64,
                prompt_tokens: 1_000,
                session_key,
                ..RequestFacts::default()
            },
            constraints,
        },
        members,
    };
    let now = Instant::now();
    let mut guard =
        DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let by = guard.decision().pick().expect("a pick").placed_by();
    let target = guard.take(take).expect("the target the host takes");
    guard.dispatched(now);
    let place = target.position as usize;
    list[place].increment_load();
    if !key.is_empty() {
        list[place].increment_routing_key_load(key);
    }
    held.push(guard);
    (place, by)
}

/// Three workers that take requests.
fn three_workers() -> Arc<[Arc<dyn Worker>]> {
    (0..3)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect()
}

/// Under a rule that keeps what it places from the profile, a stage hears
/// the issue and the dispatch of a request it placed, and neither of one
/// the rule placed alone: a pin of the key followed, a key the rule's own
/// assignment places, a pin at its cap that assignment places again, a
/// second target of such a pick that its host takes, and a pin the whole
/// decision follows for a request its host holds to places of its list.
/// Under the rule's default word the stage hears every one of those. The
/// session table hears each dispatch under either word: the key's next
/// request follows the worker the dispatch went to, also where that
/// dispatch moved the pin.
///
/// A gateway's sticky map keeps a request it places from the gateway's
/// policy in the same way (`model_gateway/src/policies/registry.rs`, lines
/// 543 to 546 for a pin, and 563 and 589 for the map's own assignment).
#[test]
fn a_stage_hears_no_issue_and_no_dispatch_of_a_request_its_rule_keeps_from_it() {
    let words = [RulePlaced::Untold, RulePlaced::Told];
    let assignments = [Assignment::Delegate, Assignment::MinLoad];
    for (kept, assignment) in words.into_iter().flat_map(|w| assignments.map(|a| (w, a))) {
        let case = format!("{kept:?}, {assignment:?}");
        let heard = Arc::new(Heard::default());
        let engine = hearing(assignment, kept, &heard);
        let list = three_workers();
        let members = PublishedList::new(Arc::clone(&list));
        let mut held = Held::new();
        // What the stage has heard by now: a request it placed, and under
        // the rule's default word every request.
        let mut told = (0, 0);
        let mut send = |asked: (&'static str, usize, bool), placed: PlacedBy, what: &str| {
            let (place, by) = sent(&engine, &list, &members, asked, &mut held);
            assert_eq!(by, placed, "{case}: {what}");
            if by == PlacedBy::Stages || kept == RulePlaced::Told {
                told = (told.0 + 1, told.1 + 1);
            }
            assert_eq!(heard.words(), told, "{case}: {what}");
            place
        };
        // The key's first request: the stages place it under `delegate`,
        // and the rule's own assignment places it otherwise.
        let own = assignment != Assignment::Delegate;
        let placed = if own {
            PlacedBy::Rule
        } else {
            PlacedBy::Stages
        };
        let first = send(("a-key", 0, false), placed, "a new key");
        // The key's pin is followed.
        assert_eq!(
            send(("a-key", 0, false), PlacedBy::Rule, "its pin"),
            first,
            "{case}"
        );
        if own {
            // Two requests of the key stand on its worker, the rule's cap:
            // the rule's own assignment places the next on another worker,
            // and that dispatch moves the pin there.
            let second = send(("a-key", 0, false), PlacedBy::Rule, "at the cap");
            assert_ne!(second, first, "{case}: at the cap");
            let next = send(("a-key", 0, false), PlacedBy::Rule, "the moved pin");
            assert_eq!(next, second, "{case}");
            // The host takes the second target of a key the rule's
            // assignment places: the request is the rule's as the first
            // target's is, and its dispatch pins the key where it went.
            let taken = send(("b-key", 1, false), PlacedBy::Rule, "a second target");
            let next = send(("b-key", 0, false), PlacedBy::Rule, "the pin it made");
            assert_eq!(next, taken, "{case}");
        }
        // A request held to places of its list is the whole decision's to
        // answer: it follows the key's pin there, and the pick says the
        // rule's word as the pin followed by one record does.
        let third = send(("c-key", 0, false), placed, "a third key");
        let held_to = send(
            ("c-key", 0, true),
            PlacedBy::Rule,
            "its pin, held to places",
        );
        assert_eq!(held_to, third, "{case}");
        // A request with no key is the stages'.
        send(("", 0, false), PlacedBy::Stages, "no key");
    }
}

/// The end of a request moves no count of a stock stage, so the end of one
/// its rule kept from the profile moves none: a stage hears that end with
/// no issue and no dispatch ahead of it. Of the stock stages three count
/// by a request's own words: the tally of sends the `least-load` scorer
/// reads, the bookings of `bucket`, and the credit `resident-work` keeps.
/// The tally and the credit move at a request the stages placed.
/// `bucket` books a worker a pair's partition lists, and a document
/// under the single plan alone has none: nothing of it moves at any word.
/// No count of the three moves at a completion or at a release, of a
/// request the profile was told of or of one it was not.
#[test]
fn the_end_of_a_request_kept_from_the_profile_moves_no_count_of_a_stock_stage() {
    let resident = PowerOfTwoMetric::ResidentTokens {
        request_tokens: 256,
    };
    let stock = [
        ("least_load", PolicySpec::preset("least_load"), true),
        ("bucket", PolicySpec::preset("bucket"), false),
        ("resident-work", PolicySpec::power_of_two(resident), true),
    ];
    for (name, document, counts_singles) in stock {
        let document = ruled(
            document.expect("a built-in preset"),
            Assignment::Delegate,
            RulePlaced::Untold,
        );
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let engine = PolicyEngine::new(spec);
        let list = three_workers();
        let members = PublishedList::new(Arc::clone(&list));
        let mut held = Held::new();
        // What every stage and every tally of the engine holds, with the
        // session table: a request with no key leaves the table alone.
        let counts = || format!("{:?}", engine.spec().engine());
        let before = counts();
        let (_, by) = sent(&engine, &list, &members, ("", 0, false), &mut held);
        assert_eq!(by, PlacedBy::Stages, "{name}");
        let moved = counts() != before;
        assert_eq!(moved, counts_singles, "{name}: a request the stages placed");
        // Three keys: the first request of each is the stages', and its
        // next follows the pin, which the rule keeps from the profile.
        for key in ["a-key", "b-key", "c-key"] {
            for placed in [PlacedBy::Stages, PlacedBy::Rule] {
                let (_, by) = sent(&engine, &list, &members, (key, 0, false), &mut held);
                assert_eq!(by, placed, "{name}: {key}");
            }
        }
        let standing = counts();
        let now = Instant::now();
        // The ends, newest first, so a kept request and one the profile
        // was told of take turns. The first and each fourth from it
        // complete with the one before them, and the others are released:
        // both kinds end both ways.
        for (nth, mut guard) in held.drain(..).rev().enumerate() {
            let served = {
                let pick = guard.decision().pick().expect("a pick");
                pick.targets.first().expect("a target").id
            };
            if matches!(nth % 4, 0 | 3) {
                let outcome = Outcome {
                    served,
                    tokens_in: 1_000,
                    tokens_out: 64,
                    ..Outcome::reported(200, TransportClass::Ok)
                };
                guard.completed(&outcome, now);
            } else {
                guard.released(ReleaseReason::ClientGone, now);
            }
            assert_eq!(counts(), standing, "{name}: end {nth}");
        }
    }
}
