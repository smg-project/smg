use std::{cell::RefCell, collections::BTreeSet, sync::Arc, thread, time::Duration};

use kv_index::{request_prefix_hashes, ChainBlockMap, ChainIndex, ContentHash, StoredBlock};
use parking_lot::Mutex;
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    load::state::{one_rank_load, TestFeed},
    test_support::no_health_check,
    worker::{ConnectionMode, WorkerLoadResponse, WorkerSpec, WorkerStatus, WorkerType},
};

use super::*;
use crate::{
    document::spec::{AppliedSpec, Assignment, PolicySpec, Recovery, RulePlaced},
    engine::{
        decision::{Pair, Pick, PlacedBy, Target, Targets},
        decision_engine::{DecisionEngine, HeldDecode, RequestMaker, WorkerLists},
        demands::DemandFacts,
        framework::{Filter, Picker, TieBand},
        list_cell::PublishedList,
        prompt_hashes::{HashSeeds, PromptHashes},
        records::worker_id,
        sticky_sessions::{Claim, DecodeVerdict, SessionRule, SessionVerdict},
        view::{FixedFacts, Identity, WireModes},
    },
    policies::consistent_hashing::tests::InOrder,
    stages::{Best, HasCapacity, StageRegistry},
    store::GatewayId,
    tracking::{
        lease::{Dispatch, EventKind, EventSet, Outcome, ReleaseReason, Said, TransportClass},
        DecisionGuard,
    },
};

mod drawn;
mod side_shadows;
mod vetoed;

mod affinity;
mod decode_again;
mod leases;
mod selection;

const MODEL: &str = "model-a";

/// One word a stage heard of a lease: what was said, the side, the
/// worker the lease books, the index it names, its session, its issue
/// instant, and the worker the word names beside the lease (the pick's
/// first target at an issue, the target at a dispatch, the worker that
/// served at a completion).
type Word = (&'static str, Side, u64, u8, Session, u64, u64);

thread_local! {
    /// The prompt size each word a stage heard on this thread stated,
    /// beside the word and its side.
    static SIZES: RefCell<Vec<(&'static str, Side, u32)>> = const { RefCell::new(Vec::new()) };
}

thread_local! {
    /// The side of each word a stage heard on this thread, and what
    /// its lease stated of its request's text: its characters, its
    /// bytes and the size made of them ([`Lease::prompt_size`]).
    static LENGTHS: RefCell<Vec<(Side, [u32; 3])>> = const { RefCell::new(Vec::new()) };
}

/// A picker that picks as [`Best`] and keeps every word it hears.
#[derive(Debug)]
struct Hears(Arc<Mutex<Vec<Word>>>);

impl Hears {
    fn heard(&self, word: &'static str, lease: &Lease, named: u64) {
        self.0.lock().push((
            word,
            lease.side,
            lease.booked.0,
            lease.target_index,
            lease.session,
            lease.issued_micros,
            named,
        ));
        SIZES.with_borrow_mut(|sizes| sizes.push((word, lease.side, lease.tokens)));
        let stated = [lease.chars, lease.bytes, lease.prompt_size()];
        LENGTHS.with_borrow_mut(|lengths| lengths.push((lease.side, stated)));
    }
}

impl Lifecycle for Hears {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        let first = decision.pick().and_then(|pick| pick.targets.first());
        self.heard("issued", lease, first.map_or(0, |first| first.id.0));
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, _: u32) {
        self.heard("dispatched", lease, target.id.0);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.heard("completed", lease, outcome.served.0);
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        self.heard("released", lease, 0);
    }

    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }
}

impl Picker for Hears {
    fn name(&self) -> &'static str {
        "hears"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        rows: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Best.pick(facts, rows, kept, scores, band, out);
    }
}

/// The stock pair document over round robin on both sides, its decode
/// side picking through a stage that keeps what it hears: the document
/// as applied, and the words.
fn hearing() -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
    hearing_with(None)
}

/// A picker of a case's own: its name in the catalog and the stage it
/// makes.
type OwnPicker = (&'static str, fn() -> Box<dyn Picker>);

/// [`hearing`], its prefill side picking through `prefill` where a case
/// names one, which is added to the catalog.
fn hearing_with(prefill: Option<OwnPicker>) -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
    hearing_under(3, prefill)
}

/// [`hearing_with`] under the generation `number`.
fn hearing_under(number: u64, prefill: Option<OwnPicker>) -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&heard);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("hears", move || Box::new(Hears(Arc::clone(&kept))))
        .expect("a free picker name");
    let mut document =
        PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
    document.generation = number;
    let decode = document.profiles.get_mut("decode");
    decode.expect("the decode profile").pick = "hears".into();
    if let Some((name, stage)) = prefill {
        catalog.picker(name, stage).expect("a free picker name");
        let side = document.profiles.get_mut("prefill");
        side.expect("the prefill profile").pick = name.into();
    }
    let digest = document.digest().expect("a digest");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    (spec.expect("a valid document"), heard)
}

/// The host's admission as two tests of a place: the prefills', then
/// the decodes'.
#[derive(Debug)]
struct Gates(fn(usize) -> bool, fn(usize) -> bool);

impl PairAdmission for Gates {
    fn has_capacity(&self, prefill: usize) -> bool {
        (self.0)(prefill)
    }

    fn has_room(&self, decode: usize) -> bool {
        (self.1)(decode)
    }
}

/// Every prefill has capacity and every decode a room.
const OPEN: Gates = Gates(|_| true, |_| true);

/// A ready record of `role` at `position` of its list.
fn row(position: u32, id: u64, role: Role) -> Candidate {
    Candidate::ready(position, WorkerId(id), role)
}

/// Prefills 1.., and decodes 101.., each in its list's order.
fn lists(prefills: u64, decodes: u64) -> (Vec<Candidate>, Vec<Candidate>) {
    let prefills = (0..prefills).map(|at| row(at as u32, at + 1, Role::Prefill));
    let decodes = (0..decodes).map(|at| row(at as u32, at + 101, Role::Decode));
    (prefills.collect(), decodes.collect())
}

fn facts() -> RequestFacts<'static> {
    RequestFacts {
        model: MODEL,
        phase: Phase::Prefill,
        prompt_tokens: 40,
        now_micros: 7_000,
        seed: 11,
        ..RequestFacts::default()
    }
}

/// A session key, so each side's session is one a case can tell.
fn keyed() -> RequestFacts<'static> {
    RequestFacts {
        session_key: Identity {
            hash: 77,
            plain: None,
        },
        ..facts()
    }
}

fn pool(rows: &[Candidate]) -> PoolFacts<'static> {
    PoolFacts {
        pool_size: rows.len() as u32,
        generation: 3,
        ..PoolFacts::default()
    }
}

fn pools<'a>(
    prefills: &'a [Candidate],
    decodes: &'a [Candidate],
    classes: &'a PartnerClasses,
) -> PairPools<'a> {
    PairPools {
        table: classes.table(),
        ..PairPools::of(prefills, decodes)
    }
}

/// The pair `decision` picks first: its prefill and its decode.
fn picked(decision: &Decision) -> (u64, u64) {
    let pick = decision.pick().expect("a pick");
    let first = pick.targets.first().expect("a target");
    (first.id.0, first.pair.expect("a pair").decode_id.0)
}

/// Every worker pairs with every worker.
fn every(prefills: usize, decodes: u32) -> PartnerClasses {
    let all: Vec<u32> = (0..decodes).collect();
    PartnerClasses::number(decodes as usize, (0..prefills).map(|_| all.as_slice()))
}

/// What a host states beside one side's list where it holds the side
/// to `places` and no veto of its own stands on a place.
fn unvetoed(places: Option<&[u32]>) -> Beside<'_> {
    Beside {
        places,
        stalled: &[],
        held: &[],
        unmet: None,
    }
}

/// The pair decided now over the two lists' records.
fn decide_over(
    spec: &AppliedSpec,
    facts: &RequestFacts<'_>,
    prefills: &[Candidate],
    decodes: &[Candidate],
    classes: &PartnerClasses,
) -> Decision {
    let free = Constraints::default();
    decide_records(
        spec.engine(),
        facts,
        (&free, [unvetoed(None); 2]),
        &pool(prefills),
        pools(prefills, decodes, classes),
        &mut Scratch::default(),
    )
}

/// One prefill and its decodes, the pick over them, and the words the
/// document's stages hear.
struct Decided {
    spec: AppliedSpec,
    heard: Arc<Mutex<Vec<Word>>>,
    decision: Decision,
}

impl Decided {
    /// The words heard since the last call.
    fn words(&self) -> Vec<Word> {
        std::mem::take(&mut *self.heard.lock())
    }
}

/// A pair decided over one prefill (worker 1) and `decodes` decodes
/// (workers 101..), every pair open.
fn decided(decodes: u64) -> Decided {
    let (spec, heard) = hearing();
    let (mut prefills, mut rows) = lists(1, decodes);
    let classes = every(1, decodes as u32);
    stamp(&mut prefills, &mut rows, &classes, &OPEN);
    let decision = decide_over(&spec, &facts(), &prefills, &rows, &classes);
    Decided {
        spec,
        heard,
        decision,
    }
}

/// A ready worker of `worker_type` at `url`.
fn worker(url: String, worker_type: WorkerType) -> Arc<dyn Worker> {
    let worker = <BasicWorkerBuilder>::from_spec(WorkerSpec {
        worker_type,
        ..WorkerSpec::new(url)
    })
    .health_config(no_health_check())
    .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// A held target that names its prefill alone: no decode of a first
/// decision is handed with it.
fn prefill_alone(prefill: WorkerId) -> Target {
    Target {
        id: prefill,
        ..Target::default()
    }
}

/// A host's two lists of ready workers as it publishes them, each with
/// its cell, under the engine of the stock pair document.
struct Host {
    engine: InProcess,
    heard: Arc<Mutex<Vec<Word>>>,
    prefills: Arc<[Arc<dyn Worker>]>,
    decodes: Arc<[Arc<dyn Worker>]>,
    lists: [PublishedList; 2],
}

/// A pair's request stated ahead, with the lists its host hands for it
/// and its snapshot of the workers' reports, where it lends one.
struct Asks<'a>(Request<'a>, PairLists<'a>, Option<Arc<LoadSnapshot>>);

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.0.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.0
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Pair(self.1)
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.2.as_ref()
    }
}

impl Host {
    fn of(prefills: usize, decodes: usize) -> Self {
        Self::under(hearing(), prefills, decodes)
    }

    /// [`Self::of`] under the document `spec`, with the words its
    /// stages keep.
    fn under(
        (spec, heard): (AppliedSpec, Arc<Mutex<Vec<Word>>>),
        prefills: usize,
        decodes: usize,
    ) -> Self {
        let side = |name: &str, worker_type, count| -> Arc<[Arc<dyn Worker>]> {
            let url = |at: usize| format!("grpc://{name}-{at}:9000");
            (0..count).map(|at| worker(url(at), worker_type)).collect()
        };
        let prefills = side("prefill", WorkerType::Prefill, prefills);
        let decodes = side("decode", WorkerType::Decode, decodes);
        Self::over((spec, heard), prefills, decodes)
    }

    /// [`Self::under`] over the two lists as their host built them.
    fn over(
        (spec, heard): (AppliedSpec, Arc<Mutex<Vec<Word>>>),
        prefills: Arc<[Arc<dyn Worker>]>,
        decodes: Arc<[Arc<dyn Worker>]>,
    ) -> Self {
        let engine = InProcess::new(spec);
        let lists = [&prefills, &decodes].map(|list| PublishedList::new(Arc::clone(list)));
        Self {
            engine,
            heard,
            prefills,
            decodes,
            lists,
        }
    }

    /// [`Self::of`] under a document whose profiles list no filter: a
    /// valid document that weighs no worker's health.
    fn unfiltered(prefills: usize, decodes: usize) -> Self {
        let mut host = Self::of(prefills, decodes);
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        document.generation = 3;
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        // The lists stand: each cell makes the new engine's value at
        // its first ask.
        host.engine = InProcess::new(spec);
        host
    }

    /// The host's keyed request for a pair over its lists, every pair
    /// open, its decode sent `later` or with its prefill.
    fn asks<'a>(&'a self, classes: &'a PartnerClasses, later: bool) -> Asks<'a> {
        let [prefill, decode] = &self.lists;
        let lists = PairLists {
            prefill: prefill.lent(),
            decode: decode.lent(),
            classes,
            admission: &OPEN,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        };
        let constraints = Constraints {
            decode_later: later,
            ..Constraints::default()
        };
        let request = Request {
            id: b"",
            facts: keyed(),
            constraints,
        };
        Asks(request, lists, None)
    }

    /// What the stages heard since the last call: each word, its side
    /// and the worker its lease books.
    fn words(&self) -> Vec<(&'static str, Side, WorkerId, u64)> {
        let heard = std::mem::take(&mut *self.heard.lock());
        let word = |word: Word| (word.0, word.1, WorkerId(word.2), word.5);
        heard.into_iter().map(word).collect()
    }

    /// One decision over the lists as they stand now, every pair open,
    /// drawing by `seed`: the pair's first, or the decode's again
    /// behind the `held` prefill, with no decode handed.
    fn decide(&self, admission: &dyn PairAdmission, seed: u64, held: Option<WorkerId>) -> Decision {
        let held = held.map(prefill_alone);
        self.decide_held(admission, seed, held.as_ref())
    }

    /// [`Self::decide`] behind the `held` target: its prefill, and the
    /// decode it names.
    fn decide_held(
        &self,
        admission: &dyn PairAdmission,
        seed: u64,
        held: Option<&Target>,
    ) -> Decision {
        self.decide_excluding(admission, seed, held, &[])
    }

    /// [`Self::decide_held`] for a request whose host names `excluded`,
    /// the workers it just failed to reach.
    fn decide_excluding(
        &self,
        admission: &dyn PairAdmission,
        seed: u64,
        held: Option<&Target>,
        excluded: &[WorkerId],
    ) -> Decision {
        let constraints = Constraints {
            excluded,
            ..Constraints::default()
        };
        self.decide_among(admission, seed, held, constraints, [None, None])
    }

    /// [`Self::decide_held`] for a request that states `constraints`
    /// and that its host holds to `places`: those of the prefill list
    /// and those of the decode list it may take.
    fn decide_among(
        &self,
        admission: &dyn PairAdmission,
        seed: u64,
        held: Option<&Target>,
        constraints: Constraints<'_>,
        places: [Option<&[u32]>; 2],
    ) -> Decision {
        let unmasked = [[&[][..]; 2]; 2];
        self.decide_vetoed(admission, (seed, held), constraints, places, unmasked)
    }

    /// [`Self::decide_among`] for a host whose standing vetoes mask
    /// `vetoes`: for the prefill list and then for the decode list,
    /// the places masked and those of them a hold stands on.
    fn decide_vetoed(
        &self,
        admission: &dyn PairAdmission,
        (seed, held): (u64, Option<&Target>),
        constraints: Constraints<'_>,
        places: [Option<&[u32]>; 2],
        vetoes: [[&[u32]; 2]; 2],
    ) -> Decision {
        let stated = RequestFacts { seed, ..facts() };
        self.decide_stated(admission, (stated, held), constraints, places, vetoes)
    }

    /// [`Self::decide_vetoed`] for a request that states `stated` whole:
    /// a retry names there the workers it tried.
    fn decide_stated(
        &self,
        admission: &dyn PairAdmission,
        (stated, held): (RequestFacts<'_>, Option<&Target>),
        constraints: Constraints<'_>,
        places: [Option<&[u32]>; 2],
        vetoes: [[&[u32]; 2]; 2],
    ) -> Decision {
        let [prefill, decode] = &self.lists;
        let classes = every(self.prefills.len(), self.decodes.len() as u32);
        let [[prefill_stalled, prefill_held], [decode_stalled, decode_held]] = vetoes;
        let lists = PairLists {
            prefill: prefill.lent(),
            decode: decode.lent(),
            classes: &classes,
            admission,
            prefill_places: places[0],
            decode_places: places[1],
            prefill_stalled,
            prefill_held,
            decode_stalled,
            decode_held,
        };
        let mut request = Request {
            id: b"",
            facts: stated,
            constraints,
        };
        let (engine, now) = (&self.engine, Instant::now());
        let decision = lists.decide(engine, None, now, &mut request, held);
        assert_eq!(request.facts.now_micros, engine.clock().micros(now));
        decision
    }

    /// The places of the pair `decision` picks first, in the prefill
    /// list and in the decode list. Each names its worker's own id.
    fn picked(&self, decision: &Decision) -> (usize, usize) {
        let pick = decision.pick().expect("a pick");
        let first = pick.targets.first().expect("a target");
        let pair = first.pair.expect("a pair");
        let (prefill, decode) = (first.position as usize, pair.decode_position as usize);
        assert_eq!(worker_id(self.prefills[prefill].as_ref()), first.id);
        assert_eq!(worker_id(self.decodes[decode].as_ref()), pair.decode_id);
        (prefill, decode)
    }

    /// [`Self::asks`] drawing by `seed`, with the snapshot of the
    /// workers' reports its host lends for the call.
    fn asks_by<'a>(
        &'a self,
        classes: &'a PartnerClasses,
        (seed, later): (u64, bool),
        loads: Option<Arc<LoadSnapshot>>,
    ) -> Asks<'a> {
        let mut stated = self.asks(classes, later);
        stated.0.facts = RequestFacts { seed, ..facts() };
        stated.2 = loads;
        stated
    }

    /// The places of the pair `stated` is first decided to, by the
    /// engine's entry for a whole decision.
    fn first(&self, stated: &Asks<'_>) -> (usize, usize) {
        let decided = self
            .engine
            .decide_under(GatewayId::LOCAL, stated, Instant::now());
        let (outcome, _) = decided
            .expect("an answer")
            .whole()
            .expect("a whole decision");
        self.picked(&outcome.decision)
    }
}

/// [`Host::of`] whose host says what each worker states for a request's
/// demands: `stated` answers for a side, the prefills' `0`, and a place
/// of that side's list.
fn stating(prefills: usize, decodes: usize, stated: impl Fn(usize, usize) -> FixedFacts) -> Host {
    let mut host = Host::of(prefills, decodes);
    let said = |side: usize, list: &Arc<[Arc<dyn Worker>]>| {
        let mut places = 0..;
        PublishedList::with_facts(Arc::clone(list), |_| {
            stated(side, places.next().expect("a place"))
        })
    };
    host.lists = [said(0, &host.prefills), said(1, &host.decodes)];
    host
}

/// A pair's constraints that state `demands` and nothing else.
fn demanding(demands: Demands) -> Constraints<'static> {
    Constraints {
        demands,
        ..Constraints::default()
    }
}

/// The places of each list that twelve first decisions of `host` name
/// for a request that states `cx`, each list held to `places`.
fn places_named(
    host: &Host,
    cx: Constraints<'_>,
    places: [Option<&[u32]>; 2],
) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let picks = (0..12).map(|seed| {
        let decision = host.decide_among(&OPEN, seed, None, cx, places);
        host.picked(&decision)
    });
    picks.unzip()
}

/// What a decision that sheds is shed for, and under which stage.
fn shed_of(decision: &Decision) -> (ShedReason, &'static str) {
    match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("not shed: {other:?}"),
    }
}

/// The side a decision that sheds names.
fn side_of(decision: &Decision) -> Option<PairSide> {
    match decision {
        Decision::Shed(shed) => shed.side,
        other => panic!("not shed: {other:?}"),
    }
}

/// A prefill is held to neither the completion's bound nor the way a
/// constraint was decided ([`Demands::held_by`]): it runs the input,
/// and the decode beside it holds the budget and generates the tokens.
/// So a prefill whose window is under the completion's bound stays a
/// candidate while a decode under it leaves, and the input's bound
/// holds both sides.
#[test]
fn a_pairs_prefill_is_held_to_neither_the_completions_bound_nor_a_constraints_way() {
    // Every worker states a window of 4,096 but the second decode,
    // which states 16,384. The first decode alone states a way to
    // decide a constraint.
    let host = stating(2, 2, |side, at| FixedFacts {
        demands: DemandFacts {
            window: if (side, at) == (1, 1) { 16_384 } else { 4_096 },
            decoding: if (side, at) == (1, 0) { 3 } else { 0 },
            ..DemandFacts::default()
        },
        ..FixedFacts::default()
    });
    let both = BTreeSet::from([0, 1]);
    let long = demanding(Demands {
        output_window: 8_000,
        ..Demands::default()
    });
    let wide = (both.clone(), BTreeSet::from([1]));
    assert_eq!(places_named(&host, long, [None, None]), wide);
    let decided = demanding(Demands {
        decoding: 3,
        ..Demands::default()
    });
    let alike = (both, BTreeSet::from([0]));
    assert_eq!(places_named(&host, decided, [None, None]), alike);
    // The input's bound is every worker's: no prefill holds it.
    let prompt = demanding(Demands {
        input_window: 8_000,
        ..Demands::default()
    });
    let decision = host.decide_among(&OPEN, 0, None, prompt, [None, None]);
    let narrow = (ShedReason::NoEndpoints, "context-window");
    assert_eq!(shed_of(&decision), narrow);
}

/// A side the demand leaves nobody sheds the pair with no endpoint
/// under the stage of the demand that took that side's last worker, as
/// a single request's list is shed, and the shed names that side.
/// Where each side is left nobody, the prefills' demand is the one
/// named, and the prefills' side.
///
/// The classic router holds a pair's pools to the request's own rule
/// and refuses for the prefill's role where no prefill is left, ahead
/// of the decodes
/// (`model_gateway/src/routers/common/placement.rs:2900-2918`, then
/// `:2943-2970`, on main `d43c7b2a9`).
#[test]
fn a_side_a_demand_leaves_nobody_sheds_the_pair_under_that_demands_stage() {
    // No prefill takes images. Every decode does, and each states a
    // window of 4,096.
    let host = stating(2, 2, |side, _| FixedFacts {
        vision: side == 1,
        demands: DemandFacts {
            window: if side == 1 { 4_096 } else { 0 },
            ..DemandFacts::default()
        },
        ..FixedFacts::default()
    });
    let shed_under = |demands: Demands| {
        let decision = host.decide_among(&OPEN, 0, None, demanding(demands), [None, None]);
        (shed_of(&decision), side_of(&decision))
    };
    let pixels = Demands {
        vision: true,
        ..Demands::default()
    };
    let blind = ((ShedReason::NoEndpoints, "vision"), Some(PairSide::Prefill));
    assert_eq!(shed_under(pixels), blind);
    let long = Demands {
        output_window: 8_000,
        ..Demands::default()
    };
    let narrow = (ShedReason::NoEndpoints, "context-window");
    assert_eq!(shed_under(long), (narrow, Some(PairSide::Decode)));
    // Both sides are left nobody, each by another demand.
    let each = Demands {
        vision: true,
        ..long
    };
    assert_eq!(shed_under(each), blind);
}

/// The stock presets of a pair as a document in force, round robin for
/// its prefills and `decode` for its decodes, with the list its stages'
/// words would be kept in.
fn presets(decode: &str) -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
    let document = PolicySpec::pair_presets("round_robin", decode);
    let mut document = document.expect("the pair presets");
    document.generation = 3;
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    (spec, Arc::default())
}

/// The first target a decision names.
fn first_target(decision: &Decision) -> Target {
    let first = decision.pick().and_then(|pick| pick.targets.first());
    *first.expect("a target")
}

fn dispatched(side: Side, index: u8) -> Said {
    let dispatch = Dispatch {
        target_index: index,
        side,
        since_decision_micros: 12,
    };
    Said::Dispatched(dispatch, Instant::now())
}

fn outcome(side: Side, served: WorkerId) -> Outcome {
    Outcome {
        side,
        served,
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    }
}

/// The pick's target at `index`.
fn target_at(decision: &Decision, index: usize) -> Target {
    let pick = decision.pick().expect("a pick");
    *pick.targets.as_slice().get(index).expect("a target")
}

/// A pair's first pick issued as a first decision issues it
/// ([`first_sides`], [`tell_first`]), with no engine above the stages
/// and so no catch.
fn issue(
    engine: &Engine,
    facts: &RequestFacts<'_>,
    decision: &Decision,
    id: LeaseId,
    later: bool,
) -> Option<(Lease, Session)> {
    let (lease, decode, session) = first_sides(facts, decision, id)?;
    let told = first_told(decision, later);
    tell_first(engine, &lease, &decode, decision, told);
    Some((lease, session))
}

/// The whole decision the leasing entry answers a host's request.
fn leased(host: &Host, maker: &impl RequestMaker, now: Instant) -> DecideOutcome {
    let decided = host.engine.decide_under(GatewayId::LOCAL, maker, now);
    decided.expect("an answer").whole().expect("no place").0
}

/// The targets a decision names, in their order: none for a shed.
fn named(decision: &Decision) -> &[Target] {
    decision.pick().map_or(&[], |pick| pick.targets.as_slice())
}

/// What a decision that picks nothing is shed for, and under which
/// stage's name.
fn shed_for(decision: &Decision) -> (ShedReason, &'static str) {
    match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("not shed: {other:?}"),
    }
}

/// A picker that picks as [`Best`] and panics at the issue of the side
/// it names.
#[derive(Debug)]
struct StopsAt(Side);

impl Lifecycle for StopsAt {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        assert!(lease.side != self.0, "this stage panics there");
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Issued])
    }
}

impl Picker for StopsAt {
    fn name(&self) -> &'static str {
        "stops-at"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        rows: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Best.pick(facts, rows, kept, scores, band, out);
    }
}

/// Four prefills and four decodes under the pair document of the two
/// presets, every pair open.
fn of_presets(prefill: &str, decode: &str) -> (Host, PartnerClasses) {
    let mut document = PolicySpec::pair_presets(prefill, decode).expect("presets");
    document.generation = 3;
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    (Host::under((spec, Arc::default()), 4, 4), every(4, 4))
}

/// Those workers' reports: every worker of a side waits on a long queue
/// but the one at the place named for its side.
fn reports(prefill: usize, decode: usize) -> Vec<(String, WorkerLoadResponse)> {
    let side = |side: &'static str, idle: usize| {
        (0..4).map(move |at| {
            let queued = if at == idle { 0 } else { 90_000 };
            let url = format!("grpc://{side}-{at}:9000");
            (url, one_rank_load(queued, 0.1, 1_000.0))
        })
    };
    let both = side("prefill", prefill).chain(side("decode", decode));
    both.collect()
}

/// A snapshot of those reports, as a request gives one.
fn queues(prefill: usize, decode: usize) -> Arc<LoadSnapshot> {
    LoadSnapshot::from_loads_for_test(reports(prefill, decode))
}

/// One side's list engine after engine, one engine a number of
/// `shape`: that many ranks of one data-parallel engine in their
/// order, or a worker that registered no rank for a `1`. Beside it,
/// each worker's engine by its place in the list.
fn ranked(
    name: &str,
    worker_type: WorkerType,
    shape: &[usize],
) -> (Arc<[Arc<dyn Worker>]>, Vec<usize>) {
    let (mut list, mut engine_of) = (Vec::new(), Vec::new());
    for (engine, &size) in shape.iter().enumerate() {
        for rank in 0..size {
            let built = <BasicWorkerBuilder>::from_spec(WorkerSpec {
                worker_type,
                ..WorkerSpec::new(format!("grpc://{name}-{engine}:9000"))
            });
            let built = match size {
                1 => built,
                _ => built.dp_config(rank, size),
            };
            let worker = built.health_config(no_health_check()).build();
            worker.set_status(WorkerStatus::Ready);
            list.push(Arc::new(worker) as Arc<dyn Worker>);
            engine_of.push(engine);
        }
    }
    (list.into(), engine_of)
}

/// A host of four prefills and four decodes under the stock policies
/// `prefill` and `decode` of a pair's two sides.
fn host_under(prefill: &str, decode: &str) -> Host {
    let document = PolicySpec::pair_presets(prefill, decode).expect("the pair presets");
    host_of(document, 4, 4)
}

/// A host of `prefills` and `decodes` workers under `document`.
fn host_of(mut document: PolicySpec, prefills: usize, decodes: usize) -> Host {
    document.generation = 3;
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    Host::under((spec, Arc::default()), prefills, decodes)
}

/// A prompt of four whole blocks as its host hashes it, its tokens
/// counted from `first`.
fn prompt(host: &Host, first: u32) -> PromptHashes {
    let tokens: Vec<u32> = (first..first + 64).collect();
    let spec = host.engine.spec();
    let seeds = HashSeeds::PLAIN;
    let hashes = PromptHashes::under_document(&spec, Some(16), Some(&tokens), None, seeds);
    let mut read = RequestFacts::default();
    hashes.write_to(&mut read);
    assert_eq!(read.block_hashes.len(), 4, "four hashed blocks");
    hashes
}

/// One decision of a pair for the prompt `hashes`, drawing by `seed`:
/// the pair's first, or the decode's again behind `held`.
fn decide_prompt(host: &Host, hashes: &PromptHashes, seed: u64, held: Option<&Target>) -> Decision {
    decide_prompt_at(host, hashes, seed, held, Instant::now())
}

/// [`decide_prompt`] at the instant `now`.
fn decide_prompt_at(
    host: &Host,
    hashes: &PromptHashes,
    seed: u64,
    held: Option<&Target>,
    now: Instant,
) -> Decision {
    let classes = every(host.prefills.len(), host.decodes.len() as u32);
    let Asks(mut request, lists, _) = host.asks(&classes, false);
    request.facts = RequestFacts { seed, ..facts() };
    hashes.write_to(&mut request.facts);
    lists.decide(&host.engine, None, now, &mut request, held)
}

/// A side reads its own list's column and no other's: with
/// `cache_aware` on the prefills and `round_robin` on the decodes, a
/// prompt's prefill stays and its decodes take their turns.
#[test]
fn a_side_that_reads_no_affinity_keeps_its_own_rule() {
    let host = host_under("cache_aware", "round_robin");
    let sent = prompt(&host, 1_000);
    let (prefill, _) = host.picked(&decide_prompt(&host, &sent, 1, None));
    let mut decodes = BTreeSet::new();
    for seed in 2..10 {
        let again = host.picked(&decide_prompt(&host, &sent, seed, None));
        assert_eq!(again.0, prefill, "seed {seed}");
        decodes.insert(again.1);
    }
    assert_eq!(decodes.len(), 4, "{decodes:?}");
}

/// Asserts that under `host` a prompt's decode stays while its prefills
/// take their turns, and that the decode asked again behind a held
/// prefill is the prompt's too.
fn the_decode_stays_and_the_prefills_turn(host: &Host) {
    let sent = prompt(host, 1_000);
    let (prefill, decode) = host.picked(&decide_prompt(host, &sent, 1, None));
    let held = prefill_alone(worker_id(host.prefills[prefill].as_ref()));
    let mut prefills = BTreeSet::from([prefill]);
    for seed in 2..10 {
        let again = host.picked(&decide_prompt(host, &sent, seed, None));
        assert_eq!(again.1, decode, "seed {seed}");
        prefills.insert(again.0);
        let behind = decide_prompt(host, &sent, seed, Some(&held));
        assert_eq!(host.picked(&behind), (prefill, decode), "seed {seed}");
    }
    assert_eq!(prefills.len(), 4, "{prefills:?}");
}

/// The mirror of the case above, where the decode alone reads: with
/// `round_robin` on the prefills and `cache_aware` on the decodes, a
/// prompt's decode stays and its prefills take their turns. The
/// decode's pick is written and none for the prefill.
#[test]
fn a_decode_that_reads_alone_keeps_its_prompts_decode() {
    the_decode_stays_and_the_prefills_turn(&host_under("round_robin", "cache_aware"));
}

/// The stock pair document over `policy` on both sides, under a
/// sessions rule with no cap: `assignment` places a new key, and
/// `kept` is the rule's word of a request it places.
fn kept_document(policy: &str, assignment: Assignment, kept: RulePlaced) -> PolicySpec {
    let mut document = PolicySpec::pair_presets(policy, policy).expect("the pair presets");
    let stock = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rule = stock.bindings[0].sessions.clone().expect("the stock rule");
    (rule.assignment, rule.rule_placed) = (assignment, kept);
    document.bindings[0].sessions = Some(rule);
    document.view.headers.extend(stock.view.headers);
    document
}

/// A host of two workers a side under [`kept_document`] over round
/// robin, its decode profile picking through a stage that keeps what
/// it hears.
fn kept_host(assignment: Assignment, kept: RulePlaced) -> Host {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let words = Arc::clone(&heard);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("hears", move || Box::new(Hears(Arc::clone(&words))))
        .expect("a free picker name");
    let mut document = kept_document("round_robin", assignment, kept);
    document.generation = 3;
    let decode = document.profiles.get_mut("decode");
    decode.expect("the decode profile").pick = "hears".into();
    let spec = AppliedSpec::validate_in(&document, &catalog, &[]);
    Host::under((spec.expect("a valid document"), heard), 2, 2)
}

/// A decode call that states a verdict of its key hands the session
/// table its pair's lease date, with a claim and with none. The side's
/// dispatch is told under that lease, and it is then written into the
/// life the call read. A decode that states no verdict hands the table
/// nothing, and neither does a lease that states no instant.
///
/// A gateway's sticky map pins a decode where that role's selection
/// picks it (`model_gateway/src/policies/manual.rs:322-368`).
#[test]
fn a_decode_calls_verdict_hands_the_table_its_pairs_lease_date() {
    let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
    let host = kept_host(Assignment::Delegate, RulePlaced::Told);
    let own = host.engine.loaded();
    let table = own.document.spec.engine().sessions();
    let rule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 0,
        ttl_micros: Some(1_000),
        sweep_micros: 100,
        recovery: Recovery::Latest,
        rule_placed: RulePlaced::Told,
        sides: Default::default(),
    };
    // The call's pick: the held prefill (worker 1) with decode 102.
    let call = |verdict: SessionVerdict| {
        let mut prefill = Targets::new();
        prefill.push(Target {
            id: id(1),
            ..Target::default()
        });
        let mut targets = Targets::new();
        targets.push(Target {
            id: id(102),
            ..Target::default()
        });
        let first = *prefill.first().expect("a prefill");
        targets.pair_under(&first, verdict, prefill.placing());
        Decision::Pick(Pick {
            targets,
            ..Pick::default()
        })
    };
    // The call decides at 5,000. Its pair's lease is of 10, or states no
    // instant. Another request began the key's life at 4,000 and was
    // served on decode 102. The call follows that pin, and its send
    // lands on decode 101.
    let rows = [
        (pinned, 10, [id(101), id(102)]),
        (SessionVerdict::Respill, 10, [id(101), id(102)]),
        (SessionVerdict::None, 10, [id(102), id(0)]),
        (pinned, 0, [id(102), id(0)]),
    ];
    for (hash, (verdict, lease, pins)) in (70..).zip(rows) {
        let facts = RequestFacts {
            now_micros: 5_000,
            session_key: Identity { hash, plain: None },
            ..facts()
        };
        let key = decode_session(&facts);
        table.lookup(key, 4_000, &rule);
        table.dispatched(key, pinned, id(102), 4_000);
        // A decode that states no verdict never asks for its key.
        let asked = Cell::new(false);
        let made = || {
            asked.set(true);
            key
        };
        let none = || (Session::NONE, WorkerId(0));
        date_decode_claim(own, &call(verdict), made, 5_000, none, lease);
        let stated = verdict != SessionVerdict::None;
        assert_eq!(asked.get(), stated, "{verdict:?} under {lease}");
        table.dispatched(key, pinned, id(101), 10);
        assert_eq!(table.pins(key), Some(pins), "{verdict:?} under {lease}");
    }
    // The call begins a key's life at 5,000 and claims decode 102. Its
    // send lands on decode 101 under the lease's date: the claim goes.
    let facts = RequestFacts {
        now_micros: 5_000,
        ..keyed()
    };
    let key = decode_session(&facts);
    let seen = table.lookup(key, 5_000, &rule);
    let made = table.claim(key, seen, id(102), 5_000);
    assert_eq!(made, Claim::Made(id(102)));
    let claimed = SessionVerdict::Claimed(id(102));
    let none = || (Session::NONE, WorkerId(0));
    date_decode_claim(own, &call(claimed), || key, 5_000, none, 10);
    table.dispatched(key, claimed, id(101), 10);
    assert_eq!(table.pins(key), Some([id(101), id(0)]), "with a claim");
}

/// Each side's record of sent prefixes keeps where a prompt went by
/// the side's own word. A key is pinned on both sides, and its next
/// request states a prompt nobody was sent: the rule follows a pin
/// that stands, and the stages place a side whose pin's worker is
/// down. The prompt then comes with no key. On a side the stages
/// placed it goes back to the worker it went to, and so it does on
/// every side under the rule's default word. On a side the rule keeps
/// from the profile the record took nothing: the prompt is placed
/// anew, and of sixteen prompts some go elsewhere.
///
/// The pick states what its prefill holds of the prompt by the prefill
/// side's word alone: under the word that keeps a request from the
/// profile it states nothing where the rule placed the prefill,
/// whatever placed the decode, and it states it where the stages placed
/// the prefill beside a decode the rule placed.
///
/// A gateway's policy is asked nothing behind a pin its sticky map
/// followed, so its record never sees that prompt
/// (`model_gateway/src/policies/registry.rs:794-797`), and each role's
/// policy keeps a record of its own
/// (`model_gateway/src/routers/common/placement.rs:2827-2828`).
#[test]
fn each_sides_record_keeps_a_prompt_by_the_sides_own_word() {
    const PROMPTS: u32 = 16;
    let downs = [None, Some(Side::Prefill), Some(Side::Decode)];
    let words = [RulePlaced::Untold, RulePlaced::Told];
    for (kept, down) in words.into_iter().flat_map(|w| downs.map(|d| (w, d))) {
        let case = format!("{kept:?}, {down:?} down");
        let document = kept_document("cache_aware", Assignment::Delegate, kept);
        let host = host_of(document, 4, 4);
        let classes = every(4, 4);
        // The places of one decision of the prompt `hashes`, under
        // `key` where one is given.
        let decide = |hashes: &PromptHashes, seed: u64, key: Option<u64>| {
            let Asks(mut request, lists, _) = host.asks(&classes, false);
            request.facts = RequestFacts {
                seed,
                session_key: key.map_or(Identity::NONE, Identity::hashed),
                ..facts()
            };
            hashes.write_to(&mut request.facts);
            let now = Instant::now();
            let decision = lists.decide(&host.engine, None, now, &mut request, None);
            let stated = decision.pick().is_some_and(|pick| pick.overlap().of != 0);
            (host.picked(&decision), stated)
        };
        // How many prompts went back to their worker, a side.
        let mut back = [0; 2];
        for at in 0..PROMPTS {
            let (key, seed) = (Some(500 + u64::from(at)), u64::from(at));
            let (pinned, _) = decide(&prompt(&host, 1_000_000 + 1_000 * at), seed, key);
            let gone = match down {
                Some(Side::Prefill) => Some(&host.prefills[pinned.0]),
                Some(Side::Decode) => Some(&host.decodes[pinned.1]),
                _ => None,
            };
            if let Some(worker) = gone {
                worker.set_status(WorkerStatus::NotReady);
            }
            let sent = prompt(&host, 2_000_000 + 1_000 * at);
            let (went, stated) = decide(&sent, seed, key);
            if let Some(worker) = gone {
                worker.set_status(WorkerStatus::Ready);
            }
            let stayed = (went.0 == pinned.0, went.1 == pinned.1);
            let stands = (down != Some(Side::Prefill), down != Some(Side::Decode));
            assert_eq!(stayed, stands, "{case}: prompt {at}");
            // The statement is the prefill side's: the stages placed
            // that side under the default word, or with its pin down.
            let prefill_told = kept == RulePlaced::Told || down == Some(Side::Prefill);
            assert_eq!(stated, prefill_told, "{case}: prompt {at}: stated");
            let (next, _) = decide(&sent, seed + 1, None);
            back[0] += u32::from(next.0 == went.0);
            back[1] += u32::from(next.1 == went.1);
        }
        for (side, back) in [(Side::Prefill, back[0]), (Side::Decode, back[1])] {
            if down == Some(side) || kept == RulePlaced::Told {
                assert_eq!(back, PROMPTS, "{case}: {side:?}, recorded");
            } else {
                assert!(back < PROMPTS, "{case}: {side:?}, kept from the record");
            }
        }
    }
}
