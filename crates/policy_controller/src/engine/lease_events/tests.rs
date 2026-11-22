//! The cases of a single request's words under a sessions rule: what a
//! stage hears of a pick by who placed it, asked through the engine's own
//! entry over a list of worker objects.

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Instant,
};

use smg_worker::{identity::basic::Worker, test_support::ready_worker};

use crate::{
    document::spec::{AppliedSpec, Assignment, PolicySpec},
    engine::{
        decision::{Decision, PlacedBy, Target, Targets},
        decision_engine::{Request, RequestMaker, WorkerLists},
        framework::{Picker, TieBand},
        policy_engine::{PolicyEngine, PoolMembers},
        view::{Candidate, Constraints, Identity, RequestFacts},
    },
    policies::random::Random,
    stages::{registry::StageRegistry, Lifecycle},
    store::GatewayId,
    tracking::{
        lease::{EventSet, Lease, Side},
        DecisionGuard,
    },
};

const MODEL: &str = "model-a";

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

/// The engine of the `manual` preset's sessions rule under `assignment`,
/// with a cap of two requests of a key on a worker, beside a profile that
/// picks with [`Hears`].
fn hearing(assignment: Assignment, heard: &Arc<Heard>) -> PolicyEngine {
    let mut stages = StageRegistry::builtin();
    let kept = Arc::clone(heard);
    let added = stages.picker("hears", move || Box::new(Hears(Arc::clone(&kept))));
    added.expect("a free name");
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    let random = PolicySpec::preset("random").expect("a built-in preset");
    document.profiles = random.profiles;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "hears".into();
    let binding = &mut document.bindings[0];
    binding.profile = Some("random".into());
    let rule = binding.sessions.as_mut().expect("the stock rule");
    rule.assignment = assignment;
    rule.inflight_cap = 2;
    document.generation = 1;
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    PolicyEngine::new(spec.expect("the stages hold the picker"))
}

/// A request stated ahead, with the list its host hands for it.
struct Asks<'a> {
    request: Request<'a>,
    members: &'a PoolMembers,
}

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members)
    }
}

/// The requests a case holds in flight.
type Held<'e> = Vec<DecisionGuard<&'e PolicyEngine>>;

/// One request of `key`, with no key where it is empty, sent to its first
/// target as its host sends it: the dispatch is said, and the worker object
/// counts the request and its key. The request stays in flight in `held`.
/// The place it went to, and who placed it.
fn sent<'e>(
    engine: &'e PolicyEngine,
    list: &[Arc<dyn Worker>],
    members: &PoolMembers,
    key: &'static str,
    held: &mut Held<'e>,
) -> (usize, PlacedBy) {
    let session_key = match key {
        "" => Identity::NONE,
        key => Identity::plain(key),
    };
    let asks = Asks {
        request: Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                seed: held.len() as u64,
                session_key,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        },
        members,
    };
    let now = Instant::now();
    let mut guard =
        DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let by = guard.decision().pick().expect("a pick").placed_by();
    let target = guard.take(0).expect("the first target");
    guard.dispatched(now);
    let place = target.position as usize;
    list[place].increment_load();
    if !key.is_empty() {
        list[place].increment_routing_key_load(key);
    }
    held.push(guard);
    (place, by)
}

/// A stage hears the issue and the dispatch of a request it placed, and
/// neither of one the binding's sessions rule placed alone: a pin of the
/// key followed, a key the rule's own assignment places, and a pin at its
/// cap that assignment places again. The session table hears each of
/// those dispatches: the key's next request follows the worker the
/// dispatch went to, also where that dispatch moved the pin.
///
/// A gateway's sticky map keeps a request it places from the gateway's
/// policy in the same way (`model_gateway/src/policies/registry.rs`, lines
/// 543 to 546 for a pin, and 562 and 589 for the map's own assignment).
#[test]
fn a_stage_hears_no_issue_and_no_dispatch_of_a_request_its_sessions_rule_placed() {
    for assignment in [Assignment::Delegate, Assignment::MinLoad] {
        let heard = Arc::new(Heard::default());
        let engine = hearing(assignment, &heard);
        let list: Arc<[Arc<dyn Worker>]> = (0..3)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect();
        let members = engine.pool_members(&list);
        let mut held = Held::new();
        let mut send = |key| sent(&engine, &list, &members, key, &mut held);
        // The key's first request: the stages place it under `delegate`,
        // and the rule's own assignment places it otherwise.
        let own = assignment != Assignment::Delegate;
        let (first, by) = send("a-key");
        if own {
            assert_eq!(
                (by, heard.words()),
                (PlacedBy::Rule, (0, 0)),
                "{assignment:?}"
            );
        } else {
            assert_eq!(
                (by, heard.words()),
                (PlacedBy::Stages, (1, 1)),
                "{assignment:?}"
            );
        }
        // The key's pin is followed: no stage hears of that request.
        let before = heard.words();
        assert_eq!(send("a-key"), (first, PlacedBy::Rule), "{assignment:?}");
        assert_eq!(heard.words(), before, "{assignment:?}: a pin followed");
        if own {
            // Two requests of the key stand on its worker, the rule's cap:
            // the rule's own assignment places the next on another worker,
            // and that dispatch moves the pin there.
            let (second, by) = send("a-key");
            assert_ne!(second, first, "{assignment:?}: at the cap");
            assert_eq!(by, PlacedBy::Rule, "{assignment:?}: at the cap");
            assert_eq!(send("a-key"), (second, PlacedBy::Rule), "{assignment:?}");
            assert_eq!(heard.words(), before, "{assignment:?}: placed at the cap");
        }
        // A request with no key is the stages'.
        let (_, by) = send("");
        assert_eq!(by, PlacedBy::Stages, "{assignment:?}: no key");
        let heard = heard.words();
        assert_eq!(
            heard,
            (before.0 + 1, before.1 + 1),
            "{assignment:?}: no key"
        );
    }
}
