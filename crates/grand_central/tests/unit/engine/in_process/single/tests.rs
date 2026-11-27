use std::{borrow::Cow, cell::Cell, collections::BTreeSet, sync::Arc, time::Duration};

use kv_index::{TenantId, TokenTree, Tree as StringTree};
use parking_lot::Mutex;
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    load::state::one_rank_load,
    model_card::ModelCard,
    test_support::{no_health_check, ready_worker},
    worker::WorkerStatus,
};

use super::{super::InProcess, *};
use crate::{
    document::spec::{
        AppliedSpec, Assignment, KeySourceSpec, PolicySpec, RoutingKeySpec, RulePlaced, ScoreStage,
        ShadowSpec, WarmupSliceParams,
    },
    engine::{
        decision::{LiftedVeto, Overlap, PlacedBy, Target, Targets, MAX_TARGETS},
        decision_engine::{
            Decided, DecisionEngine, DecisionRefused, TextKey, TokenKey, WorkerLists,
        },
        demands::{DemandFacts, Demands},
        framework::{Picker, Place, PoolWord, Scorer, TieBand, DRAW_FROM},
        list_cell::PublishedList,
        prompt_hashes::HashSeeds,
        records::worker_id,
        view::{
            AffinityFacts, Candidate, Constraints, FixedFacts, Identity, RequestFacts,
            SessionCountedUnder, WireModes, WorkerId,
        },
    },
    policies::random::Random,
    stages::registry::StageRegistry,
    store::GatewayId,
    tracking::{
        lease::{Membership, Outcome, ReleaseReason, TransportClass},
        DecisionGuard,
    },
};

mod affinity;
mod demand;
mod lifecycle;
mod retry;
mod selection;

pub(crate) const MODEL: &str = "model-a";

fn preset(name: &str) -> InProcess {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    InProcess::new(AppliedSpec::validate(&document).expect("the document validates"))
}

/// `document` with a binding ahead of its first that states a sessions
/// rule for the models `ruled-*`: the rule of the preset `manual`, its
/// keys placed by the binding's own stages. The document's session
/// table takes a lease's words, whatever its stages take.
pub(crate) fn ruled_ahead(mut document: PolicySpec) -> PolicySpec {
    let manual = PolicySpec::preset("manual").expect("a built-in preset");
    document.view.headers.clone_from(&manual.view.headers);
    let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
    rule.assignment = Assignment::Delegate;
    let mut ruled = document.bindings[0].clone();
    ruled.matcher.model = "ruled-*".to_owned();
    ruled.sessions = Some(rule);
    document.bindings.insert(0, ruled);
    document
}

pub(crate) fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect()
}

/// A host that keeps a token tree of the prompts it sent to its workers
/// and lends it. It keeps no index of cache events.
struct Lends(Arc<TokenTree>);

impl overlap::CacheIndexes for Lends {
    fn index(&self, _: &str) -> Option<Arc<dyn overlap::CacheIndex>> {
        None
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        None
    }

    fn token_tree(&self, _: &str, _: Phase) -> Option<Arc<TokenTree>> {
        Some(Arc::clone(&self.0))
    }
}

/// A host that keeps a string tree of the prompts it sent as text and
/// lends it. It keeps no token tree and no index of cache events.
struct LendsText(Arc<StringTree>);

impl overlap::CacheIndexes for LendsText {
    fn index(&self, _: &str) -> Option<Arc<dyn overlap::CacheIndex>> {
        None
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        None
    }

    fn string_tree(&self, _: &str, _: Phase) -> Option<Arc<StringTree>> {
        Some(Arc::clone(&self.0))
    }
}

/// The engine of `spec` under a host that lends its string tree, and
/// the host's tree.
pub(crate) fn lending_text(spec: AppliedSpec) -> (InProcess, Arc<StringTree>) {
    let engine = InProcess::new(spec);
    let tree = Arc::new(StringTree::new());
    engine
        .affinity()
        .set_indexes(Arc::new(LendsText(Arc::clone(&tree))));
    (engine, tree)
}

/// What a lending host's document holds beyond its preset.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Also {
    /// Nothing.
    Nothing,
    /// The binding names a copy of its profile as its shadow.
    Shadow,
    /// The profile costs each worker by a named signal too. No report
    /// states it, so every worker costs the same.
    Signal,
}

/// The engine of a preset under a host that lends its token tree, a
/// page of sixteen tokens, and the host's tree. `also` is what the
/// document holds beyond the preset.
pub(crate) fn lent(name: &str, also: Also) -> (InProcess, Arc<TokenTree>) {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    let live = document.bindings[0].profile.clone();
    let live = live.expect("a binding of one profile");
    match also {
        Also::Nothing => {}
        Also::Shadow => {
            let twin = document.profiles[&live].clone();
            document.profiles.insert("twin".into(), twin);
            document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
        }
        Also::Signal => {
            let tokens = serde_json::from_str(r#"{"unit":"tokens"}"#).expect("a signal");
            document.signals.insert("held".into(), tokens);
            let cost =
                r#"{"least-cost":{"terms":{"signal.held":1},"unreported":"missing-as-zero"}}"#;
            let profile = document.profiles.get_mut(&live);
            let profile = profile.expect("the binding's profile");
            profile
                .score
                .push(serde_json::from_str(cost).expect("a scorer"));
        }
    }
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    lending(spec)
}

/// The engine of `spec` under a host that lends its token tree, a page
/// of sixteen tokens, and the host's tree: for a case that makes its
/// own document.
pub(crate) fn lending(spec: AppliedSpec) -> (InProcess, Arc<TokenTree>) {
    let engine = InProcess::new(spec);
    let tree = Arc::new(TokenTree::with_config(16, Default::default()));
    engine
        .affinity()
        .set_indexes(Arc::new(Lends(Arc::clone(&tree))));
    (engine, tree)
}

pub(crate) fn request<'a>() -> Request<'a> {
    Request {
        id: b"request",
        facts: RequestFacts {
            model: MODEL,
            seed: 7,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    }
}

/// A request stated ahead, with the lists its host hands for it, and how
/// many times the engine made it.
pub(crate) struct Asks<'a> {
    pub(crate) request: Request<'a>,
    pub(crate) members: WorkerLists<'a>,
    pub(crate) loads: Option<Arc<LoadSnapshot>>,
    pub(crate) made: Cell<u32>,
    /// The token ids the host names as the request's entry in a token
    /// tree, and how many times the engine asked for the entry.
    pub(crate) key: Option<&'a [u32]>,
    pub(crate) keyed: Cell<u32>,
    /// The text the host names as the request's entry in a string
    /// tree, and how many times the engine asked for that entry.
    pub(crate) text: Option<&'a str>,
    pub(crate) worded: Cell<u32>,
}

impl<'a> Asks<'a> {
    pub(crate) fn of(request: Request<'a>, members: Option<&'a PublishedList>) -> Self {
        let lists = match members {
            Some(list) => WorkerLists::Single(list.lent()),
            None => WorkerLists::None,
        };
        Self::over(request, lists)
    }

    fn over(request: Request<'a>, members: WorkerLists<'a>) -> Self {
        Self {
            request,
            members,
            loads: None,
            made: Cell::new(0),
            key: None,
            keyed: Cell::new(0),
            text: None,
            worded: Cell::new(0),
        }
    }
}

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.made.set(self.made.get() + 1);
        self.request
    }

    fn plain(&self) -> Option<Plain> {
        self.request.plain()
    }

    fn prompt_chars(&self) -> u32 {
        self.request.prompt_chars()
    }

    fn prompt_bytes(&self) -> u32 {
        self.request.prompt_bytes()
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.members
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads.as_ref()
    }

    fn token_key(&self, _: usize) -> Option<TokenKey<'_>> {
        self.keyed.set(self.keyed.get() + 1);
        let tokens = Cow::Borrowed(self.key?);
        Some(TokenKey { tokens, marker: 0 })
    }

    fn text_key(&self) -> Option<TextKey<'_>> {
        self.worded.set(self.worded.get() + 1);
        let text = Cow::Borrowed(self.text?);
        Some(TextKey { text, marker: 0 })
    }
}

/// What the one entry answers a request stated ahead.
pub(crate) fn answered(engine: &InProcess, asks: &Asks<'_>) -> Decided {
    let decided = engine.decide_under(GatewayId::LOCAL, asks, Instant::now());
    decided.expect("an answer")
}

/// The whole decision the entry answers: no place.
pub(crate) fn decided(engine: &InProcess, asks: &Asks<'_>) -> DecideOutcome {
    answered(engine, asks).whole().expect("a whole decision").0
}

/// Runs `run`, a decision for `asks`, as the engine's own placing call
/// runs one: the road ahead of a request's making is open to it, and a
/// decision on that road draws from the seed `asks` states. A decision
/// outside it is a host's, whose request is made.
pub(crate) fn placing<T>(asks: &Asks<'_>, run: impl FnOnce() -> T) -> T {
    use crate::engine::in_process::tests::seed_hook;
    seed_hook::stated(asks.request.facts.seed, run)
}

/// The whole decision the entry answers the placing call ([`placing`]).
pub(crate) fn ahead(engine: &InProcess, asks: &Asks<'_>) -> DecideOutcome {
    placing(asks, || decided(engine, asks))
}

/// The place the entry answers: no whole decision.
fn placed(engine: &InProcess, asks: &Asks<'_>) -> AlonePick {
    match answered(engine, asks) {
        Decided::Alone(pick) => pick,
        Decided::Whole(decided, _) => panic!("a place, not {decided:?}"),
    }
}

/// The entry reads its host's clock where a decision is dated and
/// nowhere else: a place reads none, and a retry's whole pass, a profile
/// that scores and a place noted for a stage that takes a word read it
/// once each.
#[test]
fn a_place_reads_no_clock_and_a_dated_decision_reads_it_once() {
    let list = pool(4);
    let tried = [worker_id(list[0].as_ref())];
    let read = |engine: &InProcess, retried: bool| {
        let members = PublishedList::new(Arc::clone(&list));
        let mut stated = request();
        if retried {
            stated.facts.tried = &tried;
        }
        let asks = Asks::of(stated, Some(&members));
        let reads = Cell::new(0_u32);
        let decided = engine.decide_at(&asks, || {
            reads.set(reads.get() + 1);
            Instant::now()
        });
        let place = matches!(decided.expect("an answer"), Decided::Alone(_));
        (place, reads.get())
    };
    assert_eq!(read(&preset("round_robin"), false), (true, 0));
    assert_eq!(read(&preset("round_robin"), true), (false, 1));
    assert_eq!(read(&preset("least_load"), false), (false, 1));
    let (noting, _) = hearing_as(false, true, EventSet::LEASE);
    assert_eq!(read(&noting, false), (false, 1));
}

/// A place whose profile scores, and whose picker reads the request, is
/// never picked ahead of its request: the request is made for every
/// decision, plain or not.
#[test]
fn a_place_that_scores_makes_its_request() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    for made in 1..=3 {
        assert!(decided(&engine, &asks).decision.pick().is_some());
        assert_eq!(asks.made.get(), made);
    }
}

/// Power of two under a limit on the age of a load report
/// (`view.max_report_age_ms`) is decided ahead of its making too: the
/// limit wraps least-load, and the wrapper reads of a request what
/// least-load reads.
/// No request is made, and the whole pass of the same request names the
/// same two workers in the same order.
#[test]
fn power_of_two_under_a_limit_on_a_reports_age_makes_no_request() {
    let mut document = PolicySpec::preset("power_of_two").expect("a built-in preset");
    document.generation = 1;
    document.view.max_report_age_ms = Some(900);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(spec);
    let members = PublishedList::new(pool(DRAW_FROM));
    for seed in 0..64 {
        let mut stated = request();
        stated.facts.seed = seed;
        let asks = Asks::of(stated, Some(&members));
        let decision = ahead(&engine, &asks).decision;
        assert_eq!(asks.made.get(), 0, "no request is made");
        let whole = engine.decide_without_lease(&asks, Instant::now());
        let whole = whole.expect("a whole decision").0;
        let places = |decision: &Decision| -> Vec<u32> {
            let pick = decision.pick().expect("a pick");
            pick.targets.iter().map(|target| target.position).collect()
        };
        assert_eq!(
            places(&whole),
            places(&decision),
            "the whole pass of {seed}"
        );
        assert_eq!(places(&decision).len(), 2, "the pick and the other draw");
    }
}

/// Power of two's plain single request is decided whole ahead of its
/// making, the pick and the other draw its two targets: no request is
/// made. Over workers of distinct waits the longest wait never leads,
/// since it loses beside whichever worker it is drawn with, and every
/// other worker does. The decision is run with the seed the request
/// states, which its trace states: the whole pass of the same request
/// made names the same two workers in the same order. Under a binding
/// that names a source of a routing key the request is made, since its
/// key is derived at its making. Under a document whose session table
/// alone takes a lease's words the decision of a request no sessions
/// rule binds is not asked ahead: the request is made and its whole
/// decision names two workers and holds no lease.
#[test]
fn power_of_two_is_decided_ahead_with_no_request_made() {
    let places = |decision: &Decision| -> Vec<u32> {
        let pick = decision.pick().expect("a pick");
        pick.targets.iter().map(|target| target.position).collect()
    };
    let members = PublishedList::new(pool(DRAW_FROM));
    let reports = (0..DRAW_FROM as i32)
        .map(|at| {
            let url = format!("http://worker-{at}:8000");
            (url, one_rank_load(10_000 * (at + 1), 0.1, 1_000.0))
        })
        .collect();
    let loads = Some(LoadSnapshot::from_loads_for_test(reports));
    // The request of one seed, as its host states it ahead.
    let stated = |seed: u64| {
        let mut request = request();
        request.facts.seed = seed;
        request
    };
    let engine = preset("power_of_two");
    let mut leads = [0_u32; DRAW_FROM];
    for seed in 0..2_048 {
        let asks = Asks {
            loads: loads.clone(),
            ..Asks::of(stated(seed), Some(&members))
        };
        let decision = ahead(&engine, &asks).decision;
        assert_eq!(asks.made.get(), 0, "no request is made");
        let drawn = places(&decision);
        assert_eq!(drawn.len(), 2, "the pick and the other draw");
        assert_eq!(decision.pick().expect("a pick").trace.seed, seed);
        leads[drawn[0] as usize] += 1;
        let whole = engine.decide_without_lease(&asks, Instant::now());
        let whole = whole.expect("a whole decision").0;
        assert_eq!(places(&whole), drawn, "the whole pass of seed {seed}");
    }
    assert_eq!(
        leads[DRAW_FROM - 1],
        0,
        "the longest wait loses beside every other"
    );
    assert!(
        leads[..DRAW_FROM - 1].iter().all(|&leads| leads > 0),
        "{leads:?}"
    );
    // Equal scores, at the size a whole pass draws ahead of its rows
    // from and over it: the tie falls as that pass draws it.
    for workers in [DRAW_FROM, 2 * DRAW_FROM] {
        let tied = PublishedList::new(pool(workers));
        for seed in 0..64 {
            let asks = Asks::of(stated(seed), Some(&tied));
            let decision = ahead(&engine, &asks).decision;
            let whole = engine.decide_without_lease(&asks, Instant::now());
            let whole = whole.expect("a whole decision").0;
            assert_eq!(places(&whole), places(&decision), "{workers} workers");
        }
    }

    let mut document = PolicySpec::preset("power_of_two").expect("a built-in preset");
    document.generation = 1;
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![KeySourceSpec {
            rid: true,
            ..KeySourceSpec::default()
        }],
        strip_lineage: false,
    });
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let keyed = InProcess::new(spec);
    let others = PublishedList::new(pool(DRAW_FROM));
    let asks = Asks::of(request(), Some(&others));
    assert_eq!(places(&ahead(&keyed, &asks).decision).len(), 2);
    assert_eq!(asks.made.get(), 1, "a request that may hold a key is made");

    // A sessions rule on another binding, and in the place of
    // least-load a host's scorer that takes no word and reads no count
    // of sends: the session table alone takes a lease's words, and it
    // is told nothing of this pick. The host's scorer has the request
    // made, as a host's own is read as one that reads the request: the
    // decision is not asked ahead, and its whole decision holds no lease.
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer("alike", || Box::new(Alike) as Box<dyn Scorer>);
    added.expect("a free name");
    let mut document = PolicySpec::preset("power_of_two").expect("a built-in preset");
    let manual = PolicySpec::preset("manual").expect("a built-in preset");
    document.view.headers.clone_from(&manual.view.headers);
    let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
    rule.assignment = Assignment::Delegate;
    let mut ruled = document.bindings[0].clone();
    ruled.matcher.model = "ruled-*".to_owned();
    ruled.sessions = Some(rule);
    document.bindings.insert(0, ruled);
    document.generation = 1;
    let profile = document.profiles.get_mut("power_of_two");
    profile.expect("the preset's profile").score = vec!["alike".into()];
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let quiet = InProcess::new(spec.expect("the stages hold the scorer"));
    assert!(quiet.in_force().hears.intersects(EventSet::LEASE));
    let asks = Asks::of(request(), Some(&others));
    let whole = ahead(&quiet, &asks);
    assert_eq!(places(&whole.decision).len(), 2, "two targets");
    assert!(whole.lease.is_none(), "the table is told nothing");
    assert_eq!(asks.made.get(), 1, "the whole pass decides it");
}

/// A request a host states through a maker of its own is made, under
/// the floor of the draws and from it up: the engine asks a whole
/// decision ahead of the making for the request of its own placing call
/// alone. The decision draws from the seed the request states, and its
/// trace states that seed.
#[test]
fn a_request_a_host_states_is_made_and_drawn_from_the_seed_it_states() {
    let engine = preset("power_of_two");
    for workers in [4, DRAW_FROM] {
        let members = PublishedList::new(pool(workers));
        for seed in 0..64 {
            let mut stated = request();
            stated.facts.seed = seed;
            let asks = Asks::of(stated, Some(&members));
            let decision = decided(&engine, &asks).decision;
            let pick = decision.pick().expect("a pick");
            assert_eq!(pick.trace.seed, seed, "{workers} workers");
            assert_eq!(asks.made.get(), 1, "the request is made");
        }
    }
}

/// A list that holds an engine of several workers is decided whole,
/// under the floor of the draws and from it up: a whole pass decides
/// such a list over its engines and asks for its reports behind the
/// making, so its decision is not asked ahead and the request is made.
/// The answer is that pass's at every seed, the targets in their
/// order, and its two targets are never two workers of one engine.
#[test]
fn a_list_that_holds_an_engine_of_several_workers_is_decided_whole() {
    let places = |decision: &Decision| -> Vec<u32> {
        let pick = decision.pick().expect("a pick");
        pick.targets.iter().map(|target| target.position).collect()
    };
    let engine = preset("power_of_two");
    // Two engines of two workers; then an engine of eight workers, one
    // of four and four workers alone, at the floor of the draws.
    for shape in [&[2, 2][..], &[8, 4, 1, 1, 1, 1]] {
        let (list, engine_of) = engines(shape);
        let members = PublishedList::new(list);
        for seed in 0..256 {
            let mut stated = request();
            stated.facts.seed = seed;
            let asks = Asks::of(stated, Some(&members));
            let decision = ahead(&engine, &asks).decision;
            assert_eq!(asks.made.get(), 1, "{shape:?}: the request is made");
            let drawn = places(&decision);
            let whole = engine.decide_without_lease(&asks, Instant::now());
            let whole = whole.expect("a whole decision").0;
            assert_eq!(places(&whole), drawn, "{shape:?}: the whole pass of {seed}");
            let [lead, other] = drawn[..] else {
                panic!("{shape:?}: seed {seed} names {drawn:?}");
            };
            assert_ne!(
                engine_of[lead as usize], engine_of[other as usize],
                "{shape:?}: seed {seed} names two workers of one engine"
            );
        }
    }
}

/// A whole pass over a list under the draw's floor draws among the
/// workers its filters keep, and the decision asked ahead of the making
/// is that pass's. So the answer is the made request's at every seed,
/// whichever bit of a worker's state refuses it and wherever in the
/// list it stands, with equal scores and with distinct waits: the whole
/// decision, which is the targets in their order and the whole trace,
/// its count of the workers kept among it, or the refusal with its
/// reason. No request is made, whether the list keeps two workers or
/// more, one or none.
#[test]
fn a_list_under_the_draws_floor_is_decided_whole_with_no_request_made() {
    use smg_worker::runtime::StallReason;
    let refuse = |worker: &dyn Worker, kind: usize| match kind {
        0 => worker.set_status(WorkerStatus::NotReady),
        1 => {
            worker.set_overloaded(true);
        }
        2 => {
            worker.set_stall(Some(StallReason::Wedged));
        }
        3 => worker.set_status(WorkerStatus::Draining),
        _ => (0..16).for_each(|_| worker.record_outcome(500)),
    };
    let engine = preset("power_of_two");
    for workers in 2..DRAW_FROM {
        // The cases: nobody refused; one worker refused by each bit, at
        // the first, a middle and the last place; two refused by two
        // bits; every worker but two, every worker but one, and every
        // worker.
        let mut cases: Vec<Vec<(usize, usize)>> = vec![vec![]];
        for kind in 0..5 {
            for place in [0, workers / 2, workers - 1] {
                cases.push(vec![(place, kind)]);
            }
            cases.push(vec![(0, kind), (workers - 1, (kind + 1) % 5)]);
        }
        cases.push((2..workers).map(|place| (place, place % 5)).collect());
        cases.push((1..workers).map(|place| (place, place % 5)).collect());
        cases.push((0..workers).map(|place| (place, place % 5)).collect());
        for (case, refused) in cases.iter().enumerate() {
            for waits in [false, true] {
                let list = pool(workers);
                for &(place, kind) in refused {
                    refuse(list[place].as_ref(), kind);
                    let state = list[place].routing_state();
                    assert!(!state.eligible(), "kind {kind} refuses");
                }
                let members = PublishedList::new(list);
                let reports = (0..workers as i32)
                    .map(|at| {
                        let url = format!("http://worker-{at}:8000");
                        (url, one_rank_load(10_000 * (at + 1), 0.1, 1_000.0))
                    })
                    .collect();
                let loads = waits.then(|| LoadSnapshot::from_loads_for_test(reports));
                for seed in 0..96 {
                    let mut stated = request();
                    stated.facts.seed = seed;
                    let asks = Asks {
                        loads: loads.clone(),
                        ..Asks::of(stated, Some(&members))
                    };
                    let drawn = ahead(&engine, &asks).decision;
                    assert_eq!(
                        asks.made.get(),
                        0,
                        "{workers} workers, case {case}, waits {waits}, seed {seed}: no request is made"
                    );
                    let whole = engine.decide_without_lease(&asks, Instant::now());
                    let whole = whole.expect("a whole decision").0;
                    // The whole decision: the targets, the whole trace
                    // with its count of the workers kept, and a
                    // refusal by its reason.
                    assert_eq!(
                        drawn, whole,
                        "{workers} workers, case {case}, waits {waits}, seed {seed}"
                    );
                }
            }
        }
    }
}

/// A scorer of a host's own build that scores every worker alike and
/// declares what least-load declares: its rows are read each alone, it
/// reads no fact of the request and refuses by no report. It takes no
/// word of a lease and reads no count of sends.
#[derive(Debug)]
struct Alike;

impl Lifecycle for Alike {}

impl Scorer for Alike {
    fn name(&self) -> &'static str {
        "alike"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn refuses_by_reports(&self) -> bool {
        false
    }
}

/// Every fact a request of the placing call states, over a list under
/// the draw's floor and at it. A request that states one a whole pass
/// decides by is not plain, and is made and decided whole, each alone:
/// a worker tried, a worker left out, a subset hint, a lifted veto, a
/// later decode, a stalled row and places of its list. A session key, an
/// affinity column and a later attempt are stated by no plain request of
/// the placing call under such a place: its binding names no source of
/// a key, no lender is asked for its profile, and its request is plain
/// on a first attempt alone. A request that
/// states every other fact at once is not made: a deadline, a tenant, a
/// priority, a limit of its output, its three sizes, the hashes of its
/// prompt, a client key, its places on a lent ring and a pair's decode
/// side's, a header, media counts and a
/// worker named for a pair's decode. Its answer is the whole pass's of
/// that request at every seed, the targets in their order and the whole
/// trace, and its lease states every word the lease of the request made
/// states but the id: no stage of such a place reads one of those
/// facts. Under a document that keeps a sample the decision is kept as
/// it was answered, and it is the answer of a document that keeps none.
#[test]
fn what_a_request_states_has_it_made_or_changes_no_answer_asked_ahead() {
    use crate::engine::view::{Header, MediaCounts};
    let engine = preset("power_of_two");
    let (tried, rows) = ([WorkerId(1)], [0_u32]);
    let (hashes, ring, decode_ring) = ([3_u64, 5, 8], [17_u64, 19], [23_u64]);
    let headers = [Header {
        name: 0,
        value: "a",
    }];
    let mut stated = request();
    stated.facts = RequestFacts {
        budget_micros: 250_000,
        tenant: Identity::hashed(5),
        priority: 3,
        streaming: true,
        max_output_tokens: 512,
        prompt_tokens: 1_000,
        prompt_chars: 4_000,
        prompt_bytes: 4_100,
        block_hashes: &hashes,
        block_size: 16,
        namespace_seed: 11,
        client_key: 13,
        ring_positions: &ring,
        decode_ring_positions: Some(&decode_ring),
        headers: &headers,
        media: MediaCounts {
            images: 1,
            audio: 0,
            video: 0,
            bytes: 2_048,
        },
        ..stated.facts
    };
    stated.constraints.target_worker = Some(1);
    stated.constraints.counted_on_decode = true;
    assert!(
        stated.plain().is_some(),
        "none of them ends a plain request"
    );
    for workers in [2, 5, DRAW_FROM - 1, DRAW_FROM] {
        let members = PublishedList::new(pool(workers));
        let reports = (0..workers as i32)
            .map(|at| {
                let url = format!("http://worker-{at}:8000");
                (url, one_rank_load(10_000 * (at + 1), 0.1, 1_000.0))
            })
            .collect();
        let loads = Some(LoadSnapshot::from_loads_for_test(reports));
        let asked = |request| Asks {
            loads: loads.clone(),
            ..Asks::of(request, Some(&members))
        };
        // Each fact a whole pass decides by, alone on a request.
        let mut beyond = [request(); 7];
        beyond[0].facts.tried = &tried;
        beyond[1].constraints.excluded = &tried;
        beyond[2].constraints.subset_hint = Some(2);
        beyond[3].constraints.overload_lifted = true;
        beyond[4].constraints.decode_later = true;
        beyond[5].constraints.stalled = &rows;
        beyond[6].constraints.places = Some(&rows);
        for (fact, request) in beyond.into_iter().enumerate() {
            let asks = asked(request);
            let entry = || engine.decide_under(GatewayId::LOCAL, &asks, Instant::now());
            let answered = placing(&asks, entry);
            assert!(
                matches!(answered, Ok(Decided::Whole(..))),
                "{workers} workers, fact {fact}: {answered:?}"
            );
            assert_eq!(asks.made.get(), 1, "{workers} workers, fact {fact}: made");
        }
        // Every other fact at once.
        for seed in 0..64 {
            let case = format!("{workers} workers, seed {seed}");
            let mut request = stated;
            request.facts.seed = seed;
            let asks = asked(request);
            let now = Instant::now();
            let entry = || engine.decide_under(GatewayId::LOCAL, &asks, now);
            let answered = placing(&asks, entry);
            let drawn = answered.expect("an answer").whole().expect("no place").0;
            assert_eq!(asks.made.get(), 0, "{case}: no request is made");
            let whole = engine.decide_without_lease(&asks, now);
            assert_eq!(drawn.decision, whole.expect("a whole decision").0, "{case}");
            // The lease of the same request made, as a host states it.
            let made = asked(request);
            let answered = engine.decide_under(GatewayId::LOCAL, &made, now);
            let whole = answered.expect("an answer").whole().expect("no place").0;
            assert_eq!(made.made.get(), 1, "{case}: the request is made");
            let lease = drawn.lease.expect("a lease").lease;
            let made = whole.lease.expect("a lease").lease;
            assert_eq!(
                Lease {
                    id: made.id,
                    ..lease
                },
                made,
                "{case}"
            );
            assert_eq!(
                (lease.tokens, lease.chars, lease.bytes),
                (1_000, 4_000, 4_100)
            );
        }
    }

    let mut document = PolicySpec::preset("power_of_two").expect("a built-in preset");
    document.generation = 1;
    document.explain.sample = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let sampling = InProcess::new(spec);
    let (kept, plain) = (PublishedList::new(pool(5)), PublishedList::new(pool(5)));
    let now = Instant::now();
    let asks = Asks::of(stated, Some(&kept));
    let answered = placing(&asks, || {
        sampling.decide_under(GatewayId::LOCAL, &asks, now)
    });
    let drawn = answered.expect("an answer").whole().expect("no place").0;
    let record = sampling
        .explained(b"request")
        .expect("the decision is kept");
    assert_eq!((record.decision, record.at), (drawn.decision, now));
    let unsampled = ahead(&engine, &Asks::of(stated, Some(&plain))).decision;
    assert_eq!(record.decision, unsampled, "a sample changes no answer");
}

/// Under a document that keeps a sample a decision asked ahead of the
/// making is kept as the made request's is, where it is no pick too. A
/// list every worker of which is not ready sheds the request: the shed
/// holds no lease and is the record found by the request's id, the one
/// the whole pass of the request made keeps at the same instant. Over a
/// list every worker of which is overloaded the two keep one record
/// too. A sample changes no answer: an engine that keeps none answers
/// the same decision, with no request made.
#[test]
fn a_shed_decided_ahead_of_its_request_is_kept_as_the_made_requests_is() {
    let sampling = || {
        let mut document = PolicySpec::preset("power_of_two").expect("a built-in preset");
        document.generation = 1;
        document.explain.sample = 1;
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        InProcess::new(spec)
    };
    let unsampled = preset("power_of_two");
    for workers in [5, DRAW_FROM] {
        for overloaded in [false, true] {
            let case = format!("{workers} workers, overloaded {overloaded}");
            let refused = || {
                let list = pool(workers);
                for worker in list.iter() {
                    if overloaded {
                        worker.set_overloaded(true);
                    } else {
                        worker.set_status(WorkerStatus::NotReady);
                    }
                }
                PublishedList::new(list)
            };
            let (first, second, third) = (refused(), refused(), refused());
            let mut stated = request();
            stated.facts.seed = 7;
            let now = Instant::now();

            // Asked ahead, as the placing call asks.
            let keeps = sampling();
            let asks = Asks::of(stated, Some(&first));
            let answered = placing(&asks, || keeps.decide_under(GatewayId::LOCAL, &asks, now));
            let ahead = answered.expect("an answer").whole().expect("no place").0;
            let kept = keeps.explained(b"request");

            // Made: a request a host states.
            let keeps = sampling();
            let asks = Asks::of(stated, Some(&second));
            let answered = keeps.decide_under(GatewayId::LOCAL, &asks, now);
            let made = answered.expect("an answer").whole().expect("no place").0;
            assert_eq!(ahead.decision, made.decision, "{case}");
            assert_eq!(kept, keeps.explained(b"request"), "{case}");

            let asks = Asks::of(stated, Some(&third));
            let plain = placing(&asks, || decided(&unsampled, &asks)).decision;
            assert_eq!(asks.made.get(), 0, "{case}: no request is made");
            assert_eq!(ahead.decision, plain, "{case}: a sample changes no answer");
            if !overloaded {
                assert!(matches!(ahead.decision, Decision::Shed(_)), "{case}");
                assert!(ahead.lease.is_none(), "{case}: a shed holds no lease");
                let record = kept.expect("the shed is kept");
                assert_eq!(record.decision, ahead.decision, "{case}");
                assert_eq!(record.at, now, "{case}");
            }
        }
    }
}

/// A place that scores by load reads each worker's report in the
/// snapshot its host hands for the decision: the worker with the
/// shortest queue is picked at every seed. Handed no snapshot, the same
/// engine reads no report, and its picks spread over the list.
#[test]
fn a_place_that_scores_by_load_reads_the_reports_its_host_hands() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let reports = (0..4)
        .map(|at| {
            let queued = if at == 2 { 0 } else { 90_000 };
            let url = format!("http://worker-{at}:8000");
            (url, one_rank_load(queued, 0.1, 1_000.0))
        })
        .collect();
    let picked_at = |loads: Option<Arc<LoadSnapshot>>, seed: u64| {
        let mut request = request();
        request.facts.seed = seed;
        let asks = Asks {
            loads,
            ..Asks::of(request, Some(&members))
        };
        let decided = decided(&engine, &asks);
        let pick = decided.decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    let snapshot = LoadSnapshot::from_loads_for_test(reports);
    for seed in 0..32 {
        assert_eq!(
            picked_at(Some(Arc::clone(&snapshot)), seed),
            2,
            "seed {seed}"
        );
    }
    let mut unread = [0_u32; 4];
    for seed in 0..64 {
        unread[picked_at(None, seed) as usize] += 1;
    }
    assert!(unread.iter().all(|&picks| picks > 0), "{unread:?}");
}

/// A request the list's cell does not answer for is made. One that states
/// a wire its list does not dial is made, and the whole pass declines it
/// under `wires`. One of a second model over the list is made too, the
/// cell holding the place of the list's model alone: where that model's
/// place is decided ahead as well, the decision is the whole pass's of
/// the request made; so is one under a binding that states a sessions
/// rule.
#[test]
fn a_wire_not_dialed_and_a_second_model_over_a_list_decided_ahead() {
    use smg_worker::{test_support::worker_with_model_aliases, worker::WorkerType};
    const SECOND: &str = "model-b";
    let both = |workers: usize| -> Arc<[Arc<dyn Worker>]> {
        (0..workers)
            .map(|at| {
                let url = format!("http://worker-{at}:8000");
                let worker = worker_with_model_aliases(&url, MODEL, &[SECOND], WorkerType::Regular);
                worker.set_status(WorkerStatus::Ready);
                worker
            })
            .collect()
    };
    let engine = preset("power_of_two");
    let members = PublishedList::new(both(4));
    // The list's first request fills its cell under the list's model.
    let first = Asks::of(request(), Some(&members));
    assert!(ahead(&engine, &first).decision.pick().is_some());
    assert_eq!(first.made.get(), 0, "the list's model is decided ahead");

    let mut wired = request();
    wired.constraints.wire_modes = WireModes::GRPC;
    let asks = Asks::of(wired, Some(&members));
    let declined = ahead(&engine, &asks).decision;
    assert_eq!(asks.made.get(), 1, "a wire the list does not dial: made");
    assert!(
        matches!(&declined, Decision::Shed(shed) if shed.emptied_by == "wires"),
        "{declined:?}"
    );

    for seed in 0..64 {
        let mut second = request();
        second.facts.model = SECOND;
        second.facts.seed = seed;
        let asks = Asks::of(second, Some(&members));
        let drawn = ahead(&engine, &asks).decision;
        assert_eq!(asks.made.get(), 1, "seed {seed}: a second model, made");
        assert!(drawn.pick().is_some(), "seed {seed}: {drawn:?}");
        let whole = engine.decide_without_lease(&asks, Instant::now());
        assert_eq!(drawn, whole.expect("a whole decision").0, "seed {seed}");
    }

    // The same list's model under a document whose first binding rules
    // the models `ruled-*` by sessions: a request of such a model is made.
    let mut document = PolicySpec::preset("power_of_two").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&ruled_ahead(document)).expect("the document validates");
    let ruling = InProcess::new(spec);
    let members = PublishedList::new(pool(4));
    let first = Asks::of(request(), Some(&members));
    assert!(ahead(&ruling, &first).decision.pick().is_some());
    assert_eq!(first.made.get(), 0, "the list's model is decided ahead");
    let mut ruled = request();
    ruled.facts.model = "ruled-b";
    let asks = Asks::of(ruled, Some(&members));
    placing(&asks, || answered(&ruling, &asks));
    assert_eq!(asks.made.get(), 1, "a second model under a rule: made");
}

/// A request stated ahead is plain where it is a first attempt that
/// asks nothing beyond a worker of its list, and says what a pick with
/// no request made needs: its phase, its wires and its size. Each thing
/// it states beyond that makes it a request the whole pass reads. Its
/// text's two lengths are asked apart, where such a pick's lease is
/// issued.
#[test]
fn a_request_stated_ahead_says_whether_it_is_plain() {
    let mut stated = request();
    stated.facts.prompt_tokens = 9;
    stated.facts.prompt_chars = 31;
    stated.facts.prompt_bytes = 62;
    stated.constraints.wire_modes = WireModes::GRPC;
    assert_eq!(
        stated.plain(),
        Some(Plain {
            phase: Phase::Single,
            wires: WireModes::GRPC,
            prompt_tokens: 9,
        })
    );
    assert_eq!((stated.prompt_chars(), stated.prompt_bytes()), (31, 62));
    let tried = [WorkerId(1)];
    let rows = [0_u32];
    let mut retried = request();
    retried.facts.tried = &tried;
    let mut missed = request();
    missed.constraints.excluded = &tried;
    let mut hinted = request();
    hinted.constraints.subset_hint = Some(2);
    let mut lifted = request();
    lifted.constraints.overload_lifted = true;
    let mut later = request();
    later.constraints.decode_later = true;
    let mut masked = request();
    masked.constraints.stalled = &rows;
    let mut held = request();
    held.constraints.places = Some(&rows);
    for request in [retried, missed, hinted, lifted, later, masked, held] {
        assert_eq!(request.plain(), None);
    }
}

/// A turn that lands on a refused worker falls to the rotation of the
/// workers the filters kept, in every whole decision and at every size:
/// the ready workers take exact shares, a request with no routing key
/// and a new session in turn, and the list's word moves once a request.
/// The gateway's classic policy asks the list's turn by place and lets
/// a refused place fall to its set's rotation
/// (`model_gateway/src/policies/round_robin.rs:151-153`, `:164-167`),
/// and its own case holds these shares over four workers
/// (`a_refused_row_falls_to_the_admitted_set_and_the_spread_stays_even` in
/// `model_gateway/tests/unit/policies/round_robin/tests.rs`).
#[test]
fn a_refused_turn_falls_to_the_kept_workers_rotation_at_every_size() {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let manual = PolicySpec::preset("manual").expect("a built-in preset");
    document.view.headers.clone_from(&manual.view.headers);
    let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
    rule.assignment = Assignment::Delegate;
    document.bindings[0].sessions = Some(rule);
    for workers in [4, DRAW_FROM, 64] {
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let engine = InProcess::new(spec);
        let list = pool(workers);
        list[1].set_status(WorkerStatus::NotReady);
        let members = PublishedList::new(Arc::clone(&list));
        let word = &engine.members_of(members.lent()).rotation.0;
        let before = word.load(Ordering::Relaxed);
        // One request a turn of the list goes to the kept workers'
        // rotation, which comes round once in as many turns of the list
        // as workers are kept.
        let requests = workers * (workers - 1);
        let keys: Vec<String> = (0..requests).map(|key| format!("key-{key}")).collect();
        let mut times = vec![0_usize; workers];
        for (turn, key) in keys.iter().enumerate() {
            let mut ask = request();
            if turn % 2 == 1 {
                ask.facts.session_key = Identity::plain(key);
            }
            let asks = Asks::of(ask, Some(&members));
            let decided = decided(&engine, &asks);
            let pick = decided.decision.pick().expect("a pick");
            times[pick.targets.first().expect("the target").position as usize] += 1;
        }
        let mut shares = vec![workers; workers];
        shares[1] = 0;
        assert_eq!(times, shares, "{workers} workers");
        let turned = word.load(Ordering::Relaxed) - before;
        assert_eq!(
            turned, requests as u64,
            "{workers} workers: a turn a request"
        );
    }
}

/// A list of ready workers engine after engine, one engine a number
/// of `shape`: that many ranks of one data-parallel engine in their
/// order, or a worker that registered no rank for a `1`. Beside it,
/// each worker's engine by its place in the list.
fn engines(shape: &[usize]) -> (Arc<[Arc<dyn Worker>]>, Vec<usize>) {
    let address = |engine: usize| format!("http://engine-{engine}:8000");
    let rank = |engine: usize, rank: usize, size: usize| -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::new(address(engine))
            .model(ModelCard::new(MODEL))
            .dp_config(rank, size)
            .health_config(no_health_check())
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    };
    let (mut list, mut engine_of) = (Vec::new(), Vec::new());
    for (engine, &size) in shape.iter().enumerate() {
        for at in 0..size {
            list.push(match size {
                1 => ready_worker(&address(engine), MODEL),
                _ => rank(engine, at, size),
            });
            engine_of.push(engine);
        }
    }
    (list.into(), engine_of)
}

/// The default policy holds a request to its places ahead of its
/// prompt's holder. A prompt is sent to a worker, and the same prompt
/// held to places that worker is outside of is handed none but them,
/// as a request with no prompt is: over a list under the draw's floor
/// and one over it.
#[test]
fn a_prompts_holder_outside_the_places_is_not_handed() {
    for workers in [8_u32, 32] {
        let engine = preset("cache_aware");
        let list = pool(workers as usize);
        let members = PublishedList::new(Arc::clone(&list));
        let tokens: Vec<u32> = (0..1_024).collect();
        let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let sent = |places: Option<&[u32]>, seed: u64, prompted: bool| {
            let mut asked = request();
            asked.facts.seed = seed;
            asked.constraints.places = places;
            if prompted {
                hashes.write_to(&mut asked.facts);
            }
            let decided = decided(&engine, &Asks::of(asked, Some(&members)));
            let pick = decided.decision.pick().expect("a pick");
            let places = pick.targets.iter().map(|target| target.position);
            places.collect::<Vec<u32>>()
        };
        let holder = sent(None, 0, true)[0];
        let others: Vec<u32> = (0..workers)
            .filter(|&place| place != holder)
            .take(2)
            .collect();
        for seed in 1..32 {
            for prompted in [true, false] {
                let named = sent(Some(&others), seed, prompted);
                assert!(
                    named.iter().all(|place| others.contains(place)),
                    "{workers} workers, seed {seed}: {named:?}"
                );
            }
        }
    }
}

/// A request held to places is handed its prompt's holder among them,
/// where more workers outside them were sent the prompt than the
/// profile's sample of sixteen. Forty workers are sent one prompt, each
/// by a request held to it alone. The prompt held to one of them, and
/// to seven workers it was never sent to, is then handed that one under
/// every seed. The classic policy narrows its workers first and takes
/// every candidate its tree matched
/// (`model_gateway/src/routers/common/placement.rs:474-484`,
/// `model_gateway/src/policies/cache_aware.rs:2541-2552`).
#[test]
fn a_held_request_is_handed_its_prompts_holder_among_its_places() {
    let engine = preset("cache_aware");
    let list = pool(64);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..1_024).collect();
    let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
    let sent = |places: &[u32], seed: u64| {
        let mut asked = request();
        asked.facts.seed = seed;
        asked.constraints.places = Some(places);
        hashes.write_to(&mut asked.facts);
        let decided = decided(&engine, &Asks::of(asked, Some(&members)));
        let pick = decided.decision.pick().expect("a pick");
        pick.targets.iter().next().expect("a target").position
    };
    for place in 0..40 {
        assert_eq!(sent(&[place], 0), place);
    }
    for seed in 1..=32 {
        let handed = sent(&[7, 40, 41, 42, 43, 44, 45, 46], seed);
        assert_eq!(handed, 7, "seed {seed}");
    }
}

/// The rule for a list whose every worker is refused steers a held
/// request among its places. Held to a worker over its thresholds, it
/// is handed that worker with the veto lifted, beside free workers
/// outside its places.
#[test]
fn a_held_request_over_its_thresholds_everywhere_is_steered_among_its_places() {
    let engine = preset("round_robin");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    list[1].set_overloaded(true);
    let mut held = request();
    held.constraints.places = Some(&[1]);
    let decided = decided(&engine, &Asks::of(held, Some(&members)));
    let pick = decided.decision.pick().expect("a pick");
    assert_eq!(pick.trace.lifted, LiftedVeto::Overload);
    assert!(pick.targets.iter().all(|target| target.position == 1));
}

/// The default policy's single request is named the worker its prompt
/// was sent to. The whole pass reads the engine's record of sent
/// prefixes, which the pick before it wrote: a prompt goes back to its
/// worker whatever the request's seed, other prompts are not held to
/// that worker, and a worker told gone holds no prompt.
#[test]
fn the_default_policy_sends_a_prompt_back_to_the_worker_it_was_sent_to() {
    let engine = preset("cache_aware");
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
    let sent = |tokens: &[u32], seed: u64| {
        let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let asks = Asks::of(asked, Some(&members));
        let decided = decided(&engine, &asks);
        let pick = decided.decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    let asked = prompt(0);
    let first = sent(&asked, 0);
    for seed in 1..32 {
        assert_eq!(sent(&asked, seed), first, "seed {seed}");
    }
    // Sixteen other prompts are placed by their seeds, over idle workers.
    let others: Vec<u32> = (1..=16)
        .map(|at| sent(&prompt(at * 100_000), u64::from(at)))
        .collect();
    assert!(others.iter().any(|&position| position != others[0]));
    // With every worker told gone before a request, its seed places it.
    let moved = (0..32).any(|seed| {
        for worker in list.iter() {
            engine.forget(worker.url());
        }
        sent(&asked, seed) != first
    });
    assert!(moved, "a forgotten worker holds no prompt");
    assert_eq!(engine.affinity().left_out(), 0);
}

/// Round robin under a sessions rule sends a key back to the worker its
/// first request was dispatched to. Every request of the binding takes
/// the whole decision, where the rule is read ahead of the profile: one
/// pick alone would turn the rotation past the pin.
#[test]
fn a_pinned_key_under_round_robin_goes_back_to_its_worker() {
    let engine = {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let manual = PolicySpec::preset("manual").expect("a built-in preset");
        document.view.headers.clone_from(&manual.view.headers);
        let rule = manual.bindings[0].sessions.clone();
        document.bindings[0].sessions = Some(rule.expect("its rule"));
        InProcess::new(AppliedSpec::validate(&document).expect("the document validates"))
    };
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    // One request under the key, sent to its first target and served
    // there: the place it went to, and whether the engine made it.
    let sent = || {
        let mut asked = request();
        asked.facts.session_key = Identity::plain("key-a");
        let asks = Asks::of(asked, Some(&members));
        let now = Instant::now();
        let mut guard =
            DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let target = guard.take(0).expect("the first target");
        guard.dispatched(now);
        guard.completed(
            &Outcome {
                side: Side::Single,
                served: target.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        (target.position, asks.made.get() > 0)
    };
    let (pin, made) = sent();
    assert!(made, "the key's first request takes the whole decision");
    for turn in 0..6 {
        assert_eq!(sent(), (pin, true), "turn {turn}");
    }
}

/// The document of the default policy's profile under the stock
/// sessions rule, a new key placed by the profile's own stages, and
/// `kept`, the rule's word of a request it places alone.
pub(crate) fn default_under_sessions(kept: RulePlaced) -> AppliedSpec {
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    let default = PolicySpec::preset("cache_aware").expect("a built-in preset");
    document.profiles = default.profiles;
    let binding = &mut document.bindings[0];
    binding.profile = Some("cache_aware".into());
    let rule = binding.sessions.as_mut().expect("the stock rule");
    rule.assignment = Assignment::Delegate;
    rule.rule_placed = kept;
    document.generation = 1;
    AppliedSpec::validate(&document).expect("the document validates")
}

/// What `engine`'s record of sent prefixes holds of the prompt `tokens`
/// for each worker of `list`, in blocks: what a decision of the same
/// lookup is named. The lookup names the deepest holders alone, so a
/// shorter copy at another worker reads as none. The read's own answer
/// picks nobody, so it writes nothing.
fn recorded(engine: &InProcess, list: &[Arc<dyn Worker>], tokens: &[u32]) -> Vec<u32> {
    let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
    let mut asked = request();
    asked.facts.now_micros = engine.clock.micros(Instant::now());
    hashes.write_to(&mut asked.facts);
    let nobody = over_none(engine.loaded(), &asked);
    let ids: Vec<WorkerId> = list
        .iter()
        .map(|worker| worker_id(worker.as_ref()))
        .collect();
    let mut held = vec![0; list.len()];
    overlap::decide(
        engine.affinity(),
        &overlap::PoolAffinity::default(),
        engine.standing_spec(),
        list,
        &ids,
        &mut asked.facts,
        None,
        |_| None,
        || None,
        |_| true,
        |facts| {
            match facts.named {
                Some(rows) => {
                    for holder in rows.holders {
                        held[holder.row as usize] = holder.predicted_blocks;
                    }
                }
                None => {
                    for (row, worker) in facts.affinity.iter().enumerate() {
                        held[row] = worker.predicted_blocks;
                    }
                }
            }
            nobody
        },
    );
    held
}

/// A prompt of 64 blocks held by the worker at `place` of eight alone,
/// as [`recorded`] reads it.
fn held_at(place: u32) -> Vec<u32> {
    let mut held = vec![0; 8];
    held[place as usize] = 64;
    held
}

/// The default policy under a sessions rule. A key's first request is
/// placed by the whole decision, and its dispatch makes the key's pin.
/// The key's next request follows that pin by one worker's record under
/// either word of the rule: the whole decision does not run, so no
/// stage is asked and no row is written for its pass. What is kept of
/// that request's prompt is by the rule's word.
///
/// Where the rule keeps the request from the profile the record takes
/// nothing of its prompt, as a gateway's own sticky map keeps a pinned
/// request from its policy. Where the request is the profile's own, the
/// rule's default, its pick goes through the lookup that names a
/// request's holders with the answer made. The record then holds the
/// prompt for the pin's worker alone, and the prompt goes back there
/// with no key. The pick is the one made under the other word but for
/// the word itself and for what it states of the pin's worker: a prompt
/// nobody was sent is a miss of its blocks, and a pick kept from the
/// profile states nothing.
///
/// A pinned worker that is not ready gives the key back to the whole
/// decision.
#[test]
fn the_default_policy_follows_a_sessions_pin_by_one_record() {
    // The pinned requests' picks under the rule's word `kept`, a key
    // each.
    let followed = |kept: RulePlaced| {
        let engine = InProcess::new(default_under_sessions(kept));
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
        // One request of `tokens` under the session key `key`, with no
        // key where it is `0`, sent to its first target and served
        // there: the place it went to and its pick.
        let sent = |tokens: &[u32], key: u64, seed: u64| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            asked.facts.session_key = Identity::hashed(key);
            hashes.write_to(&mut asked.facts);
            let asks = Asks::of(asked, Some(&members));
            let now = Instant::now();
            let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                .expect("a decision");
            let pick = *guard.decision().pick().expect("a pick");
            let target = guard.take(0).expect("the first target");
            guard.dispatched(now);
            guard.completed(
                &Outcome {
                    side: Side::Single,
                    served: target.id,
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                },
                now,
            );
            (target.position, pick)
        };
        let fresh = |key: u32| prompt(key * 100_000 + 50_000);
        let (mut pins, mut picks) = (Vec::new(), Vec::new());
        for key in 1..=6_u32 {
            // The key's first request: the whole decision places it by
            // its seed, over idle workers, and claims it.
            let seed = u64::from(key);
            let (pin, pick) = sent(&prompt(key * 100_000), seed, seed);
            let claimed = matches!(pick.session_verdict, SessionVerdict::Claimed(_));
            assert!(claimed, "{kept:?}: key {key}");
            let whole = !pick.trace.drew_first();
            assert!(
                whole,
                "{kept:?}: key {key}: a new key is the whole decision's"
            );
            // Its next request, of a prompt nobody was sent: the pin,
            // followed by its one worker's record.
            let (went, pick) = sent(&fresh(key), seed, 1);
            let answered = (went, pick.session_verdict, pick.placed_by());
            let pinned = (pin, SessionVerdict::Pinned, PlacedBy::Rule);
            assert_eq!(answered, pinned, "{kept:?}: key {key}");
            let whole = !pick.trace.drew_first();
            assert!(!whole, "{kept:?}: key {key}: the whole decision ran");
            // What the record holds of the pinned request's prompt.
            let held = recorded(&engine, &list, &fresh(key));
            match kept {
                RulePlaced::Untold => assert_eq!(held, [0; 8], "key {key}: kept"),
                RulePlaced::Told => assert_eq!(held, held_at(pin), "key {key}: told"),
            }
            pins.push(pin);
            picks.push(pick);
        }
        assert!(
            pins.iter().any(|pin| *pin != pins[0]),
            "{kept:?}: the keys are pinned to more workers than one: {pins:?}"
        );
        if kept == RulePlaced::Told {
            // A recorded prompt goes back to its pin's worker with no
            // key.
            for key in 1..=6_u32 {
                let (went, _) = sent(&fresh(key), 0, u64::from(key) + 40);
                assert_eq!(went, pins[key as usize - 1], "key {key}: no key");
            }
        }
        // A pin whose worker is not ready is the whole decision's to
        // place anew.
        list[pins[0] as usize].set_status(WorkerStatus::NotReady);
        let (moved, pick) = sent(&prompt(100_000), 1, 3);
        let whole = !pick.trace.drew_first();
        assert!(whole, "{kept:?}: the whole decision places the key anew");
        assert_ne!(moved, pins[0], "{kept:?}");
        picks
    };
    let (kept, own) = (followed(RulePlaced::Untold), followed(RulePlaced::Told));
    // The hand-over returns the keyed entry's answer. The first key's
    // pinned request met both engines alike: the pick that went through
    // the lookup and the one that did not differ by the rule's word,
    // and by what the lookup states behind the told one. Nobody was
    // sent that prompt: a miss of its 64 blocks.
    assert_eq!(kept[0].overlap(), Overlap::NONE, "kept from the profile");
    let mut twin = kept[0];
    twin.targets.by_rule(RulePlaced::Told);
    twin.targets
        .state(Overlap::held(AffinityFacts::default(), 64));
    assert_eq!(twin, own[0]);
    assert_ne!(kept[0], own[0], "the rule's word is on the pick");
}

/// A retry over a list where an engine holds several workers is its
/// own instance of the decision ([`retried`]), and it follows its key's
/// pin by one worker's record as a first attempt does: the pin's worker
/// is no rank of the engine the request tried. What is kept of the
/// retry's prompt is by the rule's word there too. Under `told` the
/// record holds it for the pin's worker alone, and under `untold` for
/// nobody.
#[test]
fn a_retrys_followed_pin_is_recorded_by_the_rules_word() {
    for kept in [RulePlaced::Untold, RulePlaced::Told] {
        let engine = InProcess::new(default_under_sessions(kept));
        // Two engines of two ranks, then two workers alone.
        let (list, engine_of) = engines(&[2, 2, 1, 1]);
        let members = PublishedList::new(Arc::clone(&list));
        let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
        // One request of `tokens` under the session key `key` that
        // tried `tried`, sent to its first target and served there: the
        // place it went to and its pick.
        let sent = |tokens: &[u32], key: u64, tried: &[WorkerId]| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = key;
            asked.facts.session_key = Identity::hashed(key);
            if !tried.is_empty() {
                asked.facts.attempt = 1;
            }
            asked.facts.tried = tried;
            hashes.write_to(&mut asked.facts);
            let asks = Asks::of(asked, Some(&members));
            let now = Instant::now();
            let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                .expect("a decision");
            let pick = *guard.decision().pick().expect("a pick");
            let target = guard.take(0).expect("the first target");
            guard.dispatched(now);
            guard.completed(
                &Outcome {
                    side: Side::Single,
                    served: target.id,
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                },
                now,
            );
            (target.position as usize, pick)
        };
        for key in 1..=6_u32 {
            let (own, session) = (key * 100_000, u64::from(key));
            let (pin, _) = sent(&prompt(own), session, &[]);
            // The key's next request tried a worker of another engine
            // than its pin's, and comes again.
            let other = engine_of.iter().position(|&of| of != engine_of[pin]);
            let tried = [worker_id(list[other.expect("another engine")].as_ref())];
            let fresh = prompt(own + 50_000);
            let (went, pick) = sent(&fresh, session, &tried);
            let answered = (went, pick.session_verdict, pick.placed_by());
            let pinned = (pin, SessionVerdict::Pinned, PlacedBy::Rule);
            assert_eq!(answered, pinned, "{kept:?}: key {key}");
            let whole = !pick.trace.drew_first();
            assert!(!whole, "{kept:?}: key {key}: the whole decision ran");
            let mut held = vec![0; list.len()];
            if kept == RulePlaced::Told {
                held[pin] = 64;
            }
            let read = recorded(&engine, &list, &fresh);
            assert_eq!(read, held, "{kept:?}: key {key}: the retry's prompt");
        }
    }
}

/// A request stated ahead by a maker that states no plain request
/// ([`RequestMaker::plain`]'s own answer), as a host that hands a
/// decision's targets to another process states none: the leasing
/// entry then makes the request and decides it whole.
struct Whole<'a>(Asks<'a>);

impl RequestMaker for Whole<'_> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.0.make(spec)
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.0.workers()
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.0.loads()
    }
}

/// A request its sessions rule places alone says so on its pick, and its
/// prompt is recorded by the rule's own word of it, in four forms. A pin
/// followed by its one worker's record. A pin the whole decision
/// follows: for a request held to places of its list, and under a
/// binding with a shadow. And a key's first request under a rule whose
/// own assignment places it behind the profile's filters, where no
/// scorer and no picker of the profile is asked.
///
/// Where the rule keeps what it places from the profile no form leaves
/// a block of the prompt in the record: a gateway's sticky map hands
/// none of them to its policy, so the policy's record never sees the
/// prompt (`model_gateway/src/policies/registry.rs:794-797` for a pin
/// under its cap, `:840-841` for the map's own assignment). Where the
/// request is the profile's own, the rule's default, every form leaves
/// the whole prompt for the session's worker and for no other, the pin
/// by one record as the pin the whole decision follows. That prompt
/// then goes back to the session's worker with no key.
///
/// The forms under a shadow run again under a profile that reads the
/// column whole, where the record's other write stands. No recorded
/// sheet runs this.
#[test]
fn a_request_its_sessions_rule_places_alone_is_recorded_by_the_rules_word() {
    let forms = [
        (
            "a pin by one record",
            false,
            false,
            Assignment::Delegate,
            false,
        ),
        (
            "a pin held to places",
            false,
            true,
            Assignment::Delegate,
            false,
        ),
        ("a shadow", true, false, Assignment::Delegate, false),
        (
            "the rule's assignment",
            false,
            false,
            Assignment::Random,
            false,
        ),
        (
            "a shadow, the column",
            true,
            false,
            Assignment::Delegate,
            true,
        ),
        (
            "the rule's, the column",
            true,
            false,
            Assignment::Random,
            true,
        ),
    ];
    let words = [RulePlaced::Untold, RulePlaced::Told];
    for (kept, form) in words.into_iter().flat_map(|w| forms.map(|f| (w, f))) {
        let (form, shadowed, held, assignment, whole) = form;
        let case = format!("{kept:?}, {form}");
        let engine = {
            let mut document = PolicySpec::preset("manual").expect("a built-in preset");
            let default = PolicySpec::preset("cache_aware").expect("a built-in preset");
            document.profiles = default.profiles;
            let binding = &mut document.bindings[0];
            binding.profile = Some("cache_aware".into());
            let rule = binding.sessions.as_mut().expect("the stock rule");
            rule.assignment = assignment;
            rule.rule_placed = kept;
            if shadowed {
                let twin = document.profiles["cache_aware"].clone();
                document.profiles.insert("twin".into(), twin);
                document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
            }
            if whole {
                // A second warm-up stage that rules by another window:
                // the live profile is named no row, reads the column
                // whole and keeps the record by its own word.
                let profile = document.profiles.get_mut("cache_aware");
                profile.expect("the preset's profile").score.push(
                    ScoreStage::WarmupSlice(WarmupSliceParams {
                        weight: 1.0,
                        secs: Some(7),
                        ..WarmupSliceParams::default()
                    })
                    .into(),
                );
            }
            document.generation = 1;
            let applied = AppliedSpec::validate(&document).expect("the document validates");
            let cap = applied.engine().affinity_rows_cap(MODEL, Phase::Single);
            assert_eq!(cap == 0, whole, "{case}: the form of the record's write");
            let record = applied.engine().pool_record_for(MODEL, Phase::Single);
            assert!(record.is_some(), "{case}: the list keeps a record");
            InProcess::new(applied)
        };
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        let every: Vec<u32> = (0..8).collect();
        let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
        // One request sent to its first target and served there, held
        // to every place of its list where `held`: the place it went
        // to, who placed it, and whether the whole decision ran.
        let sent = |tokens: &[u32], key: u64, seed: u64, held: bool| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            asked.facts.session_key = Identity::hashed(key);
            if held {
                asked.constraints.places = Some(&every);
            }
            hashes.write_to(&mut asked.facts);
            let asks = Asks::of(asked, Some(&members));
            let now = Instant::now();
            let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                .expect("a decision");
            let pick = *guard.decision().pick().expect("a pick");
            let target = guard.take(0).expect("the first target");
            guard.dispatched(now);
            guard.completed(
                &Outcome {
                    side: Side::Single,
                    served: target.id,
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                },
                now,
            );
            (target.position, pick.placed_by(), !pick.trace.drew_first())
        };
        for key in 1..=6_u32 {
            let seed = u64::from(key);
            // The key's first request, then one of a prompt nobody was
            // sent: the rule alone places the one this form is about.
            let (first, by, _) = sent(&prompt(key * 100_000), seed, seed, false);
            let fresh = prompt(key * 100_000 + 50_000);
            let asked = if assignment == Assignment::Delegate {
                assert_eq!(by, PlacedBy::Stages, "{case}: the stages place a new key");
                let (went, by, whole) = sent(&fresh, seed, 1, held);
                assert_eq!((went, by), (first, PlacedBy::Rule), "{case}: key {key}");
                // The whole decision follows the pin of a request held
                // to places, and every pin under a shadow.
                assert_eq!(whole, shadowed || held, "{case}: key {key}");
                fresh
            } else {
                assert_eq!(by, PlacedBy::Rule, "{case}: key {key}");
                prompt(key * 100_000)
            };
            // What the record holds of the prompt the rule placed.
            let kept_of = recorded(&engine, &list, &asked);
            match kept {
                RulePlaced::Untold => assert_eq!(kept_of, [0; 8], "{case}: key {key}"),
                RulePlaced::Told => assert_eq!(kept_of, held_at(first), "{case}: key {key}"),
            }
            // The same prompt with no key: the policy places it, on the
            // session's worker where the record holds it there.
            let (went, by, _) = sent(&asked, 0, seed + 40, false);
            assert_eq!(by, PlacedBy::Stages, "{case}: no key");
            if kept == RulePlaced::Told {
                assert_eq!(went, first, "{case}: key {key}: no key");
            }
        }
    }
}

/// A conversation whose host lends its token tree and names its entry
/// there goes back to its worker however long it grows, whatever a
/// turn's seed: the tree matches the whole prompt. The record of hashed
/// prefixes holds a prompt's first 256 blocks, under the share a match
/// needs once the prompt is past 13,653 tokens. The host is asked for
/// the entry once a decision, and a worker its host took out of the tree
/// holds no prompt. A binding that names a shadow is answered the same:
/// its live decision is made inside the tree as well. So is a request
/// whose profile reads a named signal: the copy that carries its list's
/// signal cells is decided inside the tree too.
#[test]
fn a_long_conversation_goes_back_to_its_worker_by_the_token_tree() {
    for also in [Also::Nothing, Also::Shadow, Also::Signal] {
        let (engine, tree) = lent("cache_aware", also);
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        let sent = |tokens: &[u32], seed: u64| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            hashes.write_to(&mut asked.facts);
            let asks = Asks {
                key: Some(tokens),
                ..Asks::of(asked, Some(&members))
            };
            let decided = decided(&engine, &asks);
            assert_eq!(asks.keyed.get(), 1, "the entry is asked for once: {also:?}");
            let pick = decided.decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position
        };
        let mut said: Vec<u32> = (0..16_000).collect();
        let first = sent(&said, 0);
        for turn in 1..32 {
            said.extend(turn * 1_000_000..turn * 1_000_000 + 16);
            assert_eq!(sent(&said, u64::from(turn)), first, "turn {turn}: {also:?}");
        }
        // The tree is the host's to keep. With every worker taken out
        // of it before a request, the request's seed places it.
        let moved = (0..32).any(|seed| {
            for worker in list.iter() {
                tree.remove_tenant_all(&TenantId::from(worker.url()));
            }
            sent(&said, seed) != first
        });
        assert!(moved, "a worker out of the tree holds no prompt: {also:?}");
    }
}

/// A conversation whose host lends no tree goes back to its worker by the
/// engine's record of sent prefixes under an index of either kind: a model
/// whose index is of the kind hash keeps the record as one of the kind
/// tree does, so a turn is placed where the turn before it was sent,
/// whatever its seed.
#[test]
fn a_conversation_goes_back_to_its_worker_under_an_index_of_either_kind() {
    use crate::document::spec::{IndexKind, IndexSpec};
    for kind in [IndexKind::Tree, IndexKind::Hash] {
        let mut document = PolicySpec::preset("cache_aware").expect("a built-in preset");
        document.generation = 1;
        document.view.boundaries = vec![800, 1_000];
        let index = IndexSpec {
            kind,
            ..IndexSpec::default()
        };
        document.index.insert("*".into(), index);
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let engine = InProcess::new(spec);
        let members = PublishedList::new(pool(8));
        let sent = |tokens: &[u32], seed: u64| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            hashes.write_to(&mut asked.facts);
            let decision = decided(&engine, &Asks::of(asked, Some(&members))).decision;
            let pick = decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position
        };
        let mut said: Vec<u32> = (0..1_024).collect();
        let first = sent(&said, 0);
        let mut back = 0;
        for turn in 1..32 {
            said.extend(turn * 1_000_000..turn * 1_000_000 + 16);
            back += u32::from(sent(&said, u64::from(turn)) == first);
        }
        assert_eq!(back, 31, "the turns that went back under {kind:?}");
    }
}

/// Under the settings of the classic hash index a conversation goes back
/// to its worker however long its prompt: the document those settings
/// become counts a holder at any boundary, so a first turn of 4,096 token
/// ids against boundaries of 800 and 1,000 is placed where it was sent,
/// whatever a turn's seed and whatever match threshold the settings name.
#[test]
fn a_long_conversation_goes_back_to_its_worker_under_the_hash_settings() {
    use crate::document::{
        routing_settings::{PolicySettings, RoutingSettings},
        spec::IndexKind,
    };
    for threshold in [None, Some(0.9)] {
        let mut policy = PolicySettings::of_label("cache_aware").expect("a policy of that name");
        let PolicySettings::CacheAware {
            cache_index,
            cache_boundaries,
            cache_threshold,
            ..
        } = &mut policy
        else {
            panic!("the cache-aware settings");
        };
        *cache_index = IndexKind::Hash;
        *cache_boundaries = vec![800, 1_000];
        if let Some(threshold) = threshold {
            *cache_threshold = threshold;
        }
        let settings = RoutingSettings {
            policy,
            ..RoutingSettings::default()
        };
        let mut document = settings.policy_document().expect("the settings translate");
        document.generation = 1;
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let engine = InProcess::new(spec);
        let members = PublishedList::new(pool(8));
        let sent = |tokens: &[u32], seed: u64| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            hashes.write_to(&mut asked.facts);
            let decision = decided(&engine, &Asks::of(asked, Some(&members))).decision;
            let pick = decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position
        };
        let mut said: Vec<u32> = (0..4_096).collect();
        let first = sent(&said, 0);
        let mut back = 0;
        for turn in 1..32 {
            said.extend(turn * 1_000_000..turn * 1_000_000 + 16);
            back += u32::from(sent(&said, u64::from(turn)) == first);
        }
        assert_eq!(back, 31, "the turns that went back at {threshold:?}");
    }
}

/// A conversation its host holds as text goes back to its worker by the
/// host's string tree, however long it grows and whatever a turn's seed,
/// as a tokenized one does by the token tree
/// ([`a_long_conversation_goes_back_to_its_worker_by_the_token_tree`]).
/// The engine's entry hands the lookup its maker's text entry: the host
/// is asked for it once a decision, and for no token entry where it
/// lends no token tree. A worker its host took out of the tree holds no
/// prompt.
#[test]
fn a_long_conversation_of_text_goes_back_to_its_worker_by_the_string_tree() {
    let mut document = PolicySpec::preset("cache_aware").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let (engine, tree) = lending_text(spec);
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let sent = |text: &str, seed: u64| {
        let hashes = engine.prompt_hashes(MODEL, None, Some(text), HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let asks = Asks {
            text: Some(text),
            ..Asks::of(asked, Some(&members))
        };
        let decided = decided(&engine, &asks);
        let entries = (asks.worded.get(), asks.keyed.get());
        assert_eq!(entries, (1, 0), "the text entry is asked for once");
        let pick = decided.decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    // A thousand windows of text, past what a host hashes of a prompt.
    let mut said = "a system prompt of some length. ".repeat(2_000);
    let first = sent(&said, 0);
    for turn in 1..32_u64 {
        said.push_str(&format!("turn {turn:059}"));
        assert_eq!(sent(&said, turn), first, "turn {turn}");
    }
    // The tree is the host's to keep. With every worker taken out of it
    // before a request, the request's seed places it.
    let moved = (0..32).any(|seed| {
        for worker in list.iter() {
            tree.remove_tenant_all(&TenantId::from(worker.url()));
        }
        sent(&said, seed) != first
    });
    assert!(moved, "a worker out of the tree holds no prompt");
}

/// A worker admitted beside older ones takes no turn of a conversation
/// where its model's workers send no cache events: the warm-up catches a
/// worker up to what the index of cache events holds, so under an index
/// that holds no block a single request warms nobody, whatever warm-up
/// window the settings name, and a turn is placed where the turn before it
/// was sent.
#[test]
fn a_conversation_stays_with_its_worker_beside_a_new_one_where_no_cache_event_came() {
    use crate::document::routing_settings::{PolicySettings, RoutingSettings};
    // A window of one second, so that seven workers are older than it
    // within the case's own time.
    let settings = RoutingSettings {
        policy: PolicySettings::of_label("cache_aware").expect("a policy of that name"),
        worker_warmup_secs: 1,
        ..RoutingSettings::default()
    };
    let mut document = settings.policy_document().expect("the settings translate");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(spec);
    let sent = |members: &PublishedList, tokens: &[u32], seed: u64| {
        let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let decision = decided(&engine, &Asks::of(asked, Some(members))).decision;
        let pick = decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    let mut list: Vec<Arc<dyn Worker>> = (0..7)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect();
    std::thread::sleep(Duration::from_millis(1_100));
    let older = PublishedList::new(Arc::from(list.clone()));
    let mut said: Vec<u32> = (0..4_096).collect();
    let first = sent(&older, &said, 0);
    // The eighth worker is admitted now, beside seven older than the window.
    list.push(ready_worker("http://worker-7:8000", MODEL));
    let members = PublishedList::new(Arc::from(list));
    let mut back = 0;
    for turn in 1..32 {
        said.extend(turn * 1_000_000..turn * 1_000_000 + 16);
        back += u32::from(sent(&members, &said, u64::from(turn)) == first);
    }
    assert_eq!(
        back, 31,
        "the turns that went back beside a worker admitted now"
    );
}

/// What a stage heard of a lease: the word, the worker the lease told
/// books (the target's own at a dispatch), and the lease's index.
pub(in crate::engine::in_process) type Heard = Arc<Mutex<Vec<(&'static str, WorkerId, u8)>>>;

/// The `random` picker, keeping what it hears of the words it names
/// (`takes`). `panics` makes its pick panic. `alone` makes it place a
/// request over a pool by itself, by the pool's own turn.
#[derive(Debug)]
struct Hears {
    heard: Heard,
    panics: bool,
    alone: bool,
    takes: EventSet,
}

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        self.takes
    }

    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        let heard = ("issued", lease.booked, lease.target_index);
        self.heard.lock().push(heard);
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
        let heard = ("dispatched", target.id, lease.target_index);
        self.heard.lock().push(heard);
    }

    fn on_completed(&self, lease: &Lease, _: &Outcome) {
        let heard = ("completed", lease.booked, lease.target_index);
        self.heard.lock().push(heard);
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        let heard = ("released", lease.booked, lease.target_index);
        self.heard.lock().push(heard);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        assert!(!self.panics, "this picker panics");
        let heard = match *change {
            Membership::Added(id) => ("joined", id, 0),
            Membership::Removed(id) => ("left", id, 0),
            // A leave of the cases' model, and one of any other.
            Membership::Left { id, models } if models == [MODEL] => ("left a model", id, 0),
            Membership::Left { id, .. } => ("left another model", id, 0),
            // The ids that stand, one word each.
            Membership::Snapshot(ids) => {
                let standing = ids.iter().map(|&id| ("standing", id, 0));
                self.heard.lock().extend(standing);
                return;
            }
            _ => return,
        };
        self.heard.lock().push(heard);
    }
}

impl Picker for Hears {
    fn name(&self) -> &'static str {
        "hears"
    }

    // A picker that places by itself refuses by no report.
    fn refuses_by_reports(&self) -> bool {
        !self.alone
    }

    fn picks_alone(&self) -> bool {
        self.alone
    }

    fn pick_alone(
        &self,
        pool: &PoolWord<'_>,
        keep: &mut dyn FnMut(usize) -> bool,
    ) -> Option<usize> {
        let turn = pool.word.fetch_add(1, Ordering::Relaxed) as usize % pool.rows.max(1);
        keep(turn).then_some(turn)
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
        assert!(!self.panics, "this picker panics");
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The engine of the `random` preset with [`Hears`] in its pick's place,
/// taking every word of a lease, and what it heard.
pub(in crate::engine::in_process) fn hearing(panics: bool) -> (InProcess, Heard) {
    hearing_as(panics, false, EventSet::LEASE)
}

/// The document of [`hearing`]'s engine, for a case that has its engine
/// by another call, and what its picker hears.
pub(in crate::engine::in_process) fn hearing_spec() -> (AppliedSpec, Heard) {
    hearing_spec_as(false, false, EventSet::LEASE)
}

/// [`hearing_spec`] under the generation `number`, for a case that
/// puts a later document in force.
pub(in crate::engine::in_process) fn hearing_spec_at(number: u64) -> (AppliedSpec, Heard) {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let spec = spec_picked_by_at(number, "hears", move || {
        Box::new(Hears {
            heard: Arc::clone(&kept),
            panics: false,
            alone: false,
            takes: EventSet::LEASE,
        })
    });
    (spec, heard)
}

/// A stage table whose picker `hears` takes every word of a lease and
/// keeps what it hears in a list of its own, and those lists in the
/// order the table built their pickers: a validation builds one for
/// each profile that names the picker. Documents validated over the
/// one table build the same stage under each name.
pub(in crate::engine::in_process) fn hearing_stages() -> (StageRegistry, Arc<Mutex<Vec<Heard>>>) {
    let built = Arc::new(Mutex::new(Vec::new()));
    let lists = Arc::clone(&built);
    let make = move || -> Box<dyn Picker> {
        let heard = Heard::default();
        lists.lock().push(Arc::clone(&heard));
        Box::new(Hears {
            heard,
            panics: false,
            alone: false,
            takes: EventSet::LEASE,
        })
    };
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("hears", make);
    added.expect("a free name");
    // The registration built one picker, to ask its name.
    built.lock().clear();
    (stages, built)
}

/// [`hearing`] with a picker that places by itself where `alone`, and
/// that names the words `takes`.
fn hearing_as(panics: bool, alone: bool, takes: EventSet) -> (InProcess, Heard) {
    let (spec, heard) = hearing_spec_as(panics, alone, takes);
    (InProcess::new(spec), heard)
}

/// The document of [`hearing_as`]'s engine, and what its picker hears.
fn hearing_spec_as(panics: bool, alone: bool, takes: EventSet) -> (AppliedSpec, Heard) {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let spec = spec_picked_by("hears", move || {
        Box::new(Hears {
            heard: Arc::clone(&kept),
            panics,
            alone,
            takes,
        })
    });
    (spec, heard)
}

/// The engine of the `random` preset with the picker `make` builds, a
/// stage of the host's own build under `name`, in its pick's place.
fn picked_by(
    name: &'static str,
    make: impl Fn() -> Box<dyn Picker> + Send + Sync + 'static,
) -> InProcess {
    InProcess::new(spec_picked_by(name, make))
}

/// The document of [`picked_by`]'s engine.
fn spec_picked_by(
    name: &'static str,
    make: impl Fn() -> Box<dyn Picker> + Send + Sync + 'static,
) -> AppliedSpec {
    spec_picked_by_at(1, name, make)
}

/// [`spec_picked_by`] under the generation `number`.
fn spec_picked_by_at(
    number: u64,
    name: &'static str,
    make: impl Fn() -> Box<dyn Picker> + Send + Sync + 'static,
) -> AppliedSpec {
    let mut stages = StageRegistry::builtin();
    let added = stages.picker(name, make);
    added.expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = number;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = name.into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    spec.expect("the stages hold the picker")
}

/// One request of `asks` decided at `now`, sent to its first target and
/// served there: that target.
fn served(engine: &InProcess, asks: &Asks<'_>, now: Instant) -> Target {
    let mut guard =
        DecisionGuard::decide_under(engine, GatewayId::LOCAL, asks, now).expect("a decision");
    let target = guard.take(0).expect("the first target");
    guard.dispatched(now);
    let outcome = Outcome {
        side: Side::Single,
        served: target.id,
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    guard.completed(&outcome, now);
    target
}

/// One model has a worker that takes images.
/// A vision request for a model with no list is shed under `none`.
#[test]
fn a_demand_for_a_model_no_member_serves_is_shed_under_no_stage() {
    let engine = preset("random");
    let members = PublishedList::with_facts(pool(1), |_| FixedFacts {
        vision: true,
        ..FixedFacts::default()
    });
    let mut asked = request();
    asked.constraints.demands.vision = true;
    let listed = decided(&engine, &Asks::of(asked, Some(&members)));
    assert!(
        listed.decision.pick().is_some(),
        "the listed model takes images"
    );

    asked.facts.model = "a-model-nobody-serves";
    let absent = decided(&engine, &Asks::of(asked, None)).decision;
    let Decision::Shed(shed) = absent else {
        panic!("expected a shed for the absent model, got {absent:?}");
    };
    assert_eq!(
        (shed.reason, shed.emptied_by),
        (ShedReason::NoEndpoints, "none")
    );
}

/// The first pick records the block for the blind worker.
/// An unrestricted repeat picks that worker again.
/// A vision request for the same block picks the sighted worker.
#[test]
fn a_vision_demand_skips_the_blind_holder_of_the_first_block() {
    let engine = preset("cache_aware");
    let list = pool(2);
    let (blind, sighted) = (&list[0], &list[1]);
    let members = PublishedList::with_facts(Arc::clone(&list), |worker| FixedFacts {
        vision: Arc::ptr_eq(worker, sighted),
        ..FixedFacts::default()
    });

    let mut prompt = request();
    prompt.facts.block_hashes = &[17];
    let mut to_blind = prompt;
    to_blind.constraints.places = Some(&[0]);
    let mut vision = prompt;
    vision.constraints.demands.vision = true;

    for (asked, expected, step) in [
        (to_blind, blind, "first placement"),
        (prompt, blind, "unrestricted repeat"),
        (vision, sighted, "vision request"),
    ] {
        let outcome = decided(&engine, &Asks::of(asked, Some(&members)));
        let pick = outcome.decision.pick().expect("a pick");
        let first = pick.targets.first().expect("a target");
        assert_eq!(first.id, worker_id(expected.as_ref()), "{step}");
    }
}

/// The picker declares its needs while the document is built.
/// Two single requests get picks without another call to `needs` or a fault.
#[test]
fn a_stage_is_asked_what_it_needs_at_the_build_alone() {
    use std::sync::atomic::AtomicBool;

    use crate::{
        document::spec::Needs,
        engine::{framework::DecisionScope, view::PoolFacts},
    };

    #[derive(Debug)]
    struct BuildNeeds(Arc<AtomicBool>);

    impl Lifecycle for BuildNeeds {}

    impl DecisionScope for BuildNeeds {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

        fn close(&self) {}
    }

    impl Picker for BuildNeeds {
        fn name(&self) -> &'static str {
            "build-needs"
        }

        fn needs(&self) -> Needs {
            assert!(!self.0.load(Ordering::Relaxed), "needs ran after the build");
            Needs {
                active_sessions: true,
                ..Needs::default()
            }
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
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

    let armed = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&armed);
    let engine = picked_by("build-needs", move || {
        Box::new(BuildNeeds(Arc::clone(&flag)))
    });
    armed.store(true, Ordering::Relaxed);

    let members = PublishedList::new(pool(3));
    for id in [b"r-1", b"r-2"] {
        let asks = Asks::of(Request { id, ..request() }, Some(&members));
        assert!(decided(&engine, &asks).decision.pick().is_some());
    }
    assert_eq!(engine.faults(), 0);
    assert!(
        engine.spec().needs().active_sessions,
        "the build recorded the picker's declared need"
    );
}

/// Under the settings of the classic hash index a worker admitted beside
/// older ones takes no turn of a conversation: the classic hash index
/// calls no warm-up slice, so the document those settings become names
/// nobody as warming, whatever warm-up window the settings name, and a
/// turn is placed where the turn before it was sent.
#[test]
fn a_conversation_stays_with_its_worker_beside_a_new_one_under_the_hash_settings() {
    use crate::document::{
        routing_settings::{PolicySettings, RoutingSettings},
        spec::IndexKind,
    };
    let mut policy = PolicySettings::of_label("cache_aware").expect("a policy of that name");
    let PolicySettings::CacheAware {
        cache_index,
        cache_boundaries,
        ..
    } = &mut policy
    else {
        panic!("the cache-aware settings");
    };
    *cache_index = IndexKind::Hash;
    *cache_boundaries = vec![800, 1_000];
    // A window of one second, so that seven workers are older than it
    // within the case's own time.
    let settings = RoutingSettings {
        policy,
        worker_warmup_secs: 1,
        ..RoutingSettings::default()
    };
    let mut document = settings.policy_document().expect("the settings translate");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(spec);
    let sent = |members: &PublishedList, tokens: &[u32], seed: u64| {
        let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let decision = decided(&engine, &Asks::of(asked, Some(members))).decision;
        let pick = decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    let mut list: Vec<Arc<dyn Worker>> = (0..7)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect();
    std::thread::sleep(Duration::from_millis(1_100));
    let older = PublishedList::new(Arc::from(list.clone()));
    let mut said: Vec<u32> = (0..4_096).collect();
    let first = sent(&older, &said, 0);
    // The eighth worker is admitted now, beside seven older than the window.
    list.push(ready_worker("http://worker-7:8000", MODEL));
    let members = PublishedList::new(Arc::from(list));
    let mut back = 0;
    for turn in 1..32 {
        said.extend(turn * 1_000_000..turn * 1_000_000 + 16);
        back += u32::from(sent(&members, &said, u64::from(turn)) == first);
    }
    assert_eq!(
        back, 31,
        "the turns that went back beside a worker admitted now"
    );
}
