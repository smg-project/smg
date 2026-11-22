//! The cases of a document applied to a running engine: each asks whole
//! decisions through the engine's one entry, over a list of worker objects
//! that a program lends with its cell.

use std::{cell::Cell, sync::Arc, time::Instant};

use parking_lot::Mutex;
use smg_worker::{
    identity::basic::Worker,
    load::state::{one_rank_load, LoadSnapshot},
    test_support::ready_worker,
};

use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, Target, Targets},
        decision_engine::{DecideOutcome, DecisionEngine, Request, RequestMaker, WorkerLists},
        framework::{Picker, TieBand},
        list_cell::{LentList, ListCell},
        policy_engine::{PolicyEngine, PoolMembers},
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, Identity, RequestFacts, WorkerId},
    },
    policies::random::Random,
    stages::{registry::StageRegistry, Lifecycle},
    store::{GatewayId, StaleGeneration},
    tracking::{
        lease::{EventSet, Lease, Membership, Outcome, ReleaseReason, Side, TransportClass},
        DecisionGuard,
    },
};

const MODEL: &str = "model-a";

/// What the stages of a case's documents heard, in order: the document a
/// stage is of, and the word.
type Heard = Arc<Mutex<Vec<(&'static str, &'static str)>>>;

/// The `random` picker under a document's name, keeping each pick it makes
/// and each word it hears of the kinds it names (`takes`).
#[derive(Debug)]
struct Hears {
    document: &'static str,
    heard: Heard,
    takes: EventSet,
}

impl Hears {
    fn keep(&self, word: &'static str) {
        self.heard.lock().push((self.document, word));
    }
}

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        self.takes
    }

    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        self.keep("issued");
    }

    fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
        self.keep("dispatched");
    }

    fn on_completed(&self, _: &Lease, _: &Outcome) {
        self.keep("completed");
    }

    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        self.keep("released");
    }

    fn on_membership(&self, change: &Membership<'_>) {
        match change {
            Membership::Added(_) => self.keep("joined"),
            Membership::Removed(_) => self.keep("left"),
            _ => {}
        }
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
        self.keep("picked");
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The `random` preset as the document `name` of `generation`, with
/// [`Hears`] in its pick's place, taking the words `takes`.
fn hearing(name: &'static str, generation: u64, takes: EventSet, heard: &Heard) -> AppliedSpec {
    let mut stages = StageRegistry::builtin();
    let kept = Arc::clone(heard);
    let added = stages.picker("hears", move || {
        Box::new(Hears {
            document: name,
            heard: Arc::clone(&kept),
            takes,
        })
    });
    added.expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = generation;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "hears".into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    spec.expect("the stages hold the picker")
}

/// A stock document of `generation`.
fn preset(name: &str, generation: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = generation;
    AppliedSpec::validate(&document).expect("the document validates")
}

fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&url(at), MODEL))
        .collect()
}

/// A request of `seed` and of the session key `key`, with no key where it
/// is `0`.
fn request<'a>(seed: u64, key: u64) -> Request<'a> {
    Request {
        id: b"request",
        facts: RequestFacts {
            model: MODEL,
            seed,
            session_key: Identity::hashed(key),
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    }
}

/// A program's worker list with the cell it keeps beside it.
struct Listed {
    workers: Arc<[Arc<dyn Worker>]>,
    cell: ListCell,
}

impl Listed {
    fn of(workers: usize) -> Self {
        Self::over(pool(workers))
    }

    /// A list the program published, with its empty cell.
    fn over(workers: Arc<[Arc<dyn Worker>]>) -> Self {
        let cell = ListCell::default();
        Self { workers, cell }
    }

    /// A request over the list as a program states it: the program asks
    /// the cell for the engine's value ahead of the decision, under the
    /// document then in force.
    fn asks<'a>(&'a self, engine: &'a PolicyEngine) -> Asks<'a> {
        let lent = LentList {
            workers: &self.workers,
            cell: &self.cell,
            facts: None,
        };
        Asks {
            engine,
            members: engine.members_of(lent),
            seed: 7,
            key: 0,
            loads: None,
            next: Cell::new(None),
        }
    }
}

/// A request over one list. Where it holds a document (`next`), it applies
/// that document to `engine` at its making: in the middle of its own
/// decision, behind the decision's entry.
struct Asks<'a> {
    engine: &'a PolicyEngine,
    members: &'a PoolMembers,
    seed: u64,
    key: u64,
    loads: Option<Arc<LoadSnapshot>>,
    next: Cell<Option<AppliedSpec>>,
}

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        if let Some(next) = self.next.take() {
            self.engine.apply(next).expect("a higher generation");
        }
        request(self.seed, self.key)
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members)
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads.as_ref()
    }
}

/// The whole decision the engine's entry answers `asks`.
fn decided(asks: &Asks<'_>) -> DecideOutcome {
    let decided = asks
        .engine
        .decide_under(GatewayId::LOCAL, asks, Instant::now());
    let whole = decided.expect("an answer").whole();
    whole.expect("a whole decision").0
}

/// The generation a whole decision's pick names, and its lease's.
fn generations(decided: &DecideOutcome) -> (u64, Option<u64>) {
    let pick = decided.decision.pick().expect("a pick");
    let lease = decided.lease.as_ref().map(|leased| leased.lease.generation);
    (pick.trace.generation, lease)
}

fn served(worker: WorkerId) -> Outcome {
    Outcome {
        side: Side::Single,
        served: worker,
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    }
}

/// What a case's stages heard since the last read.
fn heard_since(heard: &Heard) -> Vec<(&'static str, &'static str)> {
    std::mem::take(&mut *heard.lock())
}

/// A decision that began before an apply ends under the document it began
/// with: that document's stages pick it and hear its issue, and its pick
/// and its lease name that document's generation. The next decision is the
/// new document's, whole.
#[test]
fn a_decision_ends_under_the_document_it_began_with() {
    let heard = Heard::default();
    let engine = PolicyEngine::new(hearing("one", 1, EventSet::LEASE, &heard));
    let listed = Listed::of(4);
    let asks = listed.asks(&engine);
    asks.next
        .set(Some(hearing("two", 2, EventSet::LEASE, &heard)));
    let began = decided(&asks);
    assert_eq!(generations(&began), (1, Some(1)));
    assert_eq!(heard_since(&heard), [("one", "picked"), ("one", "issued")]);
    let next = decided(&listed.asks(&engine));
    assert_eq!(generations(&next), (2, Some(2)));
    assert_eq!(heard_since(&heard), [("two", "picked"), ("two", "issued")]);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The number only rises: a document whose generation is not above the one
/// in force is refused by naming the one in force, and the one in force
/// goes on deciding. A document that names no generation is among them.
#[test]
fn a_generation_not_above_the_one_in_force_is_refused() {
    let heard = Heard::default();
    let engine = PolicyEngine::new(hearing("one", 2, EventSet::NONE, &heard));
    let listed = Listed::of(4);
    for offered in [0, 1, 2] {
        let refused = engine.apply(hearing("two", offered, EventSet::NONE, &heard));
        let standing = 2;
        assert_eq!(refused, Err(StaleGeneration { standing, offered }));
        assert_eq!(generations(&decided(&listed.asks(&engine))), (2, None));
        assert_eq!(heard_since(&heard), [("one", "picked")]);
    }
    assert_eq!(
        engine.apply(hearing("two", 3, EventSet::NONE, &heard)),
        Ok(())
    );
    assert_eq!(generations(&decided(&listed.asks(&engine))), (3, None));
    assert_eq!(heard_since(&heard), [("two", "picked")]);
    let refused = engine.apply(hearing("three", 2, EventSet::NONE, &heard));
    let (standing, offered) = (3, 2);
    assert_eq!(refused, Err(StaleGeneration { standing, offered }));
}

/// The words a document takes are read with the document: a request is
/// issued a lease under a document that takes a word of one, and none under
/// a document that takes none, whichever came first.
#[test]
fn the_words_a_document_takes_follow_the_document() {
    let heard = Heard::default();
    let engine = PolicyEngine::new(hearing("one", 1, EventSet::NONE, &heard));
    let listed = Listed::of(4);
    let answered = || generations(&decided(&listed.asks(&engine)));
    assert_eq!(answered(), (1, None));
    let takes = hearing("two", 2, EventSet::LEASE, &heard);
    engine.apply(takes).expect("a higher generation");
    assert_eq!(answered(), (2, Some(2)));
    let none = hearing("three", 3, EventSet::NONE, &heard);
    engine.apply(none).expect("a higher generation");
    assert_eq!(answered(), (3, None));
    let words = [
        ("one", "picked"),
        ("two", "picked"),
        ("two", "issued"),
        ("three", "picked"),
    ];
    assert_eq!(heard_since(&heard), words);
}

/// A lease of the document before is told to no stage: the stages of the
/// document in force heard no issue of it, and hear no dispatch, no end and
/// no release of it. A lease the document in force issued is told whole.
#[test]
fn a_lease_of_the_document_before_is_told_to_no_stage() {
    let heard = Heard::default();
    let engine = PolicyEngine::new(hearing("one", 1, EventSet::LEASE, &heard));
    let listed = Listed::of(4);
    let now = Instant::now();
    let hold = || {
        let asks = listed.asks(&engine);
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
    };
    let mut sent = hold().expect("a decision");
    let unsent = hold().expect("a decision");
    let before = [
        ("one", "picked"),
        ("one", "issued"),
        ("one", "picked"),
        ("one", "issued"),
    ];
    assert_eq!(heard_since(&heard), before);
    let two = hearing("two", 2, EventSet::LEASE, &heard);
    engine.apply(two).expect("a higher generation");
    let target = sent.take(0).expect("the first target");
    sent.dispatched(now);
    sent.completed(&served(target.id), now);
    drop(sent);
    drop(unsent);
    assert_eq!(heard_since(&heard), [], "a lease of the document before");
    let mut held = hold().expect("a decision");
    let target = held.take(0).expect("the first target");
    held.dispatched(now);
    held.completed(&served(target.id), now);
    drop(held);
    let whole = [
        ("two", "picked"),
        ("two", "issued"),
        ("two", "dispatched"),
        ("two", "completed"),
    ];
    assert_eq!(heard_since(&heard), whole);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A change of membership is told to the stages of the document in force,
/// and a new document's stages are told no member at the apply.
#[test]
fn membership_is_told_to_the_document_in_force() {
    let heard = Heard::default();
    let engine = PolicyEngine::new(hearing("one", 1, EventSet::NONE, &heard));
    engine.joined(&url(0));
    engine.forget(&url(0));
    assert_eq!(heard_since(&heard), [("one", "joined"), ("one", "left")]);
    let two = hearing("two", 2, EventSet::NONE, &heard);
    engine.apply(two).expect("a higher generation");
    assert_eq!(heard_since(&heard), [], "no member is told at the apply");
    engine.joined(&url(1));
    assert_eq!(heard_since(&heard), [("two", "joined")]);
}

/// A new document's stages are told no member at the apply, and a stage
/// that keeps members needs none: its ring is born empty and learns its
/// members from the list a decision is handed. Over a list whose worker
/// left before the apply, every key names under the applied document the
/// workers that an engine started with that document names.
#[test]
fn a_stage_that_keeps_members_decides_right_behind_an_apply() {
    let named = |engine: &PolicyEngine, listed: &Listed, key: u64| {
        let mut asks = listed.asks(engine);
        (asks.seed, asks.key) = (key, key);
        let decided = decided(&asks);
        let pick = decided.decision.pick().expect("a pick");
        let targets = pick.targets.iter();
        (
            pick.trace.generation,
            targets.map(|target| target.id).collect::<Vec<_>>(),
        )
    };
    for ring in ["consistent_hashing", "prefix_hash"] {
        let applied = PolicyEngine::new(preset(ring, 1));
        let started = PolicyEngine::new(preset(ring, 2));
        // The first document's ring learns all eight workers.
        let whole = Listed::of(8);
        for key in 1..=64 {
            assert_eq!(named(&applied, &whole, key).0, 1, "{ring}: key {key}");
        }
        // The fourth leaves: its host says so, and publishes the others.
        applied.forget(&url(3));
        let others = whole.workers.iter().enumerate().filter(|&(at, _)| at != 3);
        let rest = Listed::over(others.map(|(_, worker)| Arc::clone(worker)).collect());
        let two = preset(ring, 2);
        applied.apply(two).expect("a higher generation");
        for key in 1..=64 {
            let (behind, never) = (named(&applied, &rest, key), named(&started, &rest, key));
            assert_eq!(behind, never, "{ring}: key {key}");
            assert_eq!(behind.0, 2, "{ring}: key {key}");
        }
        assert_eq!((applied.faults(), applied.unheard()), (0, 0), "{ring}");
    }
}

/// A session's pin outlives the document it was made under: the engine's
/// own table is handed to the document applied, with its keys. A key pinned
/// at a dispatch under the document before is followed under the next one,
/// by one worker's record and with no pass over the list.
#[test]
fn a_sessions_pin_is_followed_under_the_document_applied() {
    let engine = PolicyEngine::new(preset("manual", 1));
    let listed = Listed::of(16);
    let keyed = || {
        let mut asks = listed.asks(&engine);
        (asks.seed, asks.key) = (41, 41);
        asks
    };
    let now = Instant::now();
    let first = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &keyed(), now);
    let mut first = first.expect("a decision");
    let placed = *first.decision().pick().expect("a pick");
    assert!(matches!(placed.session_verdict, SessionVerdict::Claimed(_)));
    let pinned = first.take(0).expect("the first target");
    first.dispatched(now);
    first.completed(&served(pinned.id), now);
    drop(first);

    let two = preset("manual", 2);
    engine.apply(two).expect("a higher generation");
    let followed = decided(&keyed());
    let pick = followed.decision.pick().expect("a pick");
    assert_eq!(pick.trace.generation, 2);
    assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
    assert_eq!(
        pick.targets.first().map(|target| target.id),
        Some(pinned.id)
    );
    assert!(pick.trace.drew_first(), "no pass over the list ran");
}

/// One report a worker, every queue as long as the next.
fn level(workers: usize) -> Arc<LoadSnapshot> {
    let reports = (0..workers).map(|at| (url(at), one_rank_load(0, 0.1, 1_000.0)));
    LoadSnapshot::from_loads_for_test(reports.collect())
}

/// The place in its list of the worker a whole decision names first.
fn place(decided: &DecideOutcome) -> u32 {
    let pick = decided.decision.pick().expect("a pick");
    pick.targets.first().expect("a target").position
}

/// A profile's count of sends passes to the next document: a request sent
/// under the document before, which no load report reflects yet, weighs on
/// its worker in the next document's decisions. An engine that sent nothing
/// places the same request on that worker again.
#[test]
fn the_sends_a_profile_counted_weigh_under_the_next_document() {
    let reports = level(4);
    let sent_under = |engine: &PolicyEngine| {
        let listed = Listed::of(4);
        let mut asks = listed.asks(engine);
        asks.loads = Some(Arc::clone(&reports));
        let now = Instant::now();
        let held = DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now);
        let mut held = held.expect("a decision");
        let target = held.take(0).expect("the first target");
        held.dispatched(now);
        // The request stays in flight: its guard is kept past the apply.
        std::mem::forget(held);
        engine
            .apply(preset("least_load", 2))
            .expect("a higher generation");
        let mut asks = listed.asks(engine);
        asks.loads = Some(Arc::clone(&reports));
        (target.position, place(&decided(&asks)))
    };
    let (first, next) = sent_under(&PolicyEngine::new(preset("least_load", 1)));
    assert_ne!(next, first, "the send weighs on its worker");
    // The same request under an engine that sent nothing.
    let idle = PolicyEngine::new(preset("least_load", 2));
    let listed = Listed::of(4);
    let mut asks = listed.asks(&idle);
    asks.loads = Some(Arc::clone(&reports));
    assert_eq!(
        place(&decided(&asks)),
        first,
        "with no send it is placed there"
    );
}

/// The document replaced is kept until the next apply and let go there, by
/// the applying thread: a decision that began under it is not the one that
/// frees it.
#[test]
fn the_document_replaced_stands_until_the_next_apply() {
    let engine = PolicyEngine::new(preset("round_robin", 1));
    let one = Arc::downgrade(&engine.document.load_full());
    engine
        .apply(preset("round_robin", 2))
        .expect("a higher generation");
    assert!(one.upgrade().is_some(), "kept past its own replacement");
    engine
        .apply(preset("round_robin", 3))
        .expect("a higher generation");
    assert!(one.upgrade().is_none(), "let go at the next apply");
}
