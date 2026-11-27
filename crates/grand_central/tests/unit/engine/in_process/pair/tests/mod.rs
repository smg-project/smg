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
        lease::{Dispatch, Outcome, ReleaseReason, Said, TransportClass},
        DecisionGuard,
    },
};

mod drawn;
mod side_shadows;
mod vetoed;

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

/// The sizes the stages heard on this thread since the last read.
fn sizes() -> Vec<(&'static str, Side, u32)> {
    SIZES.with_borrow_mut(std::mem::take)
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

/// [`hearing`]'s document and the same text under the next generation,
/// both validated over one catalog: an engine keeps the stages of the
/// first under the second.
fn hearing_and_its_next() -> ([AppliedSpec; 2], Arc<Mutex<Vec<Word>>>) {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&heard);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("hears", move || Box::new(Hears(Arc::clone(&kept))))
        .expect("a free picker name");
    let under = |number: u64| {
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets");
        document.generation = number;
        let decode = document.profiles.get_mut("decode");
        decode.expect("the decode profile").pick = "hears".into();
        let spec = AppliedSpec::validate_in(&document, &catalog, &[]);
        spec.expect("a valid document")
    };
    ([under(3), under(4)], heard)
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

#[test]
fn the_capacity_filters_name_is_the_one_a_full_list_is_read_by() {
    assert_eq!(HasCapacity.name(), HAS_CAPACITY);
}

#[test]
fn a_stamp_writes_each_records_class_and_its_sides_admission_bit() {
    let (mut prefills, mut decodes) = lists(3, 2);
    let classes = PartnerClasses::number(2, [&[0u32][..], &[1][..]]);
    let gates = Gates(|at| at != 1, |at| at == 1);
    stamp(&mut prefills, &mut decodes, &classes, &gates);
    let class = |rows: &[Candidate]| rows.iter().map(|row| row.pair_class).collect::<Vec<_>>();
    // The third prefill stands past the classes: it pairs with nobody.
    assert_eq!(class(&prefills), [1, 2, 0]);
    assert_eq!(class(&decodes), [1, 2]);
    let capacity: Vec<bool> = prefills.iter().map(|row| row.has_capacity).collect();
    assert_eq!(capacity, [true, false, true]);
    let room: Vec<bool> = decodes.iter().map(|row| row.has_room).collect();
    assert_eq!(room, [false, true]);
}

/// Two groups that never pair: whichever prefill the rotation lands
/// on, its decode is one of its own group.
#[test]
fn a_pair_is_decided_among_the_partners_the_classes_name() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(2, 4);
    let classes = PartnerClasses::number(4, [&[0u32, 1][..], &[2, 3][..]]);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let mut seen = [false; 2];
    for seed in 0..16 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
        let (prefill, decode) = picked(&decision);
        let group = if decode <= 102 { 1 } else { 2 };
        assert_eq!(group, prefill, "prefill {prefill} with decode {decode}");
        seen[prefill as usize - 1] = true;
    }
    assert_eq!(seen, [true; 2], "the rotation reaches both prefills");
}

/// The capacity bit is the host's own read: a full prefill takes no
/// request while another has room, and a list of full prefills is not
/// a refusal but a request that waits.
#[test]
fn a_full_prefill_is_passed_over_and_a_list_of_full_prefills_reads_as_full() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(2, 1);
    let classes = every(2, 1);
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|at| at == 1, |_| true),
    );
    for seed in 0..4 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
        assert_eq!(picked(&decision), (2, 101));
        assert!(!prefills_full(&decision));
    }
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|_| false, |_| true),
    );
    let decision = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
    assert!(prefills_full(&decision), "{decision:?}");
}

/// The room bit is soft: a decode with no room is passed over while
/// another has one, and where none has the list stands and the
/// dispatch waits at its pick's gate.
#[test]
fn a_decode_with_no_room_is_passed_over_and_stands_where_none_has_one() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(1, 2);
    let classes = every(1, 2);
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|_| true, |at| at == 1),
    );
    for seed in 0..4 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_over(&spec, &facts, &prefills, &decodes, &classes);
        assert_eq!(picked(&decision), (1, 102));
    }
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|_| true, |_| false),
    );
    let none_has = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
    assert_eq!(picked(&none_has).0, 1, "the list stands: {none_has:?}");
}

#[test]
fn a_request_that_states_what_a_pair_is_not_narrowed_by_is_shed_under_that_stage() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(1, 1);
    let classes = every(1, 1);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let shed_under = |cx: &Constraints<'_>| {
        let decision = decide_records(
            spec.engine(),
            &facts(),
            (cx, [unvetoed(None); 2]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        );
        match decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        }
    };
    let demanded = Constraints {
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..Constraints::default()
    };
    assert_eq!(shed_under(&demanded), (ShedReason::NoEndpoints, WIRES));
    let none = Constraints {
        subset_hint: Some(0),
        ..Constraints::default()
    };
    assert_eq!(shed_under(&none), (ShedReason::NoEndpoints, SUBSET_HINT));
    let some = Constraints {
        subset_hint: Some(2),
        ..Constraints::default()
    };
    let declined = (ShedReason::PolicyDeclined, SUBSET_HINT);
    assert_eq!(shed_under(&some), declined);
    let held = Constraints {
        places: Some(&[0]),
        ..Constraints::default()
    };
    let declined = (ShedReason::PolicyDeclined, single::PLACES);
    assert_eq!(shed_under(&held), declined);
}

/// The refusal names the side of the list that holds no record, the
/// prefills' first: the gateway's own pair selection refuses a pool
/// with no worker for that pool's role, and reads the prefills first
/// (`model_gateway/src/routers/common/placement.rs:2860-2867`, on main
/// `d43c7b2a9`).
#[test]
fn one_side_alone_is_no_pair() {
    let (spec, _) = hearing();
    let sides = [
        (2, 0, PairSide::Decode),
        (0, 2, PairSide::Prefill),
        (0, 0, PairSide::Prefill),
    ];
    for (prefills, decodes, side) in sides {
        let classes = every(prefills as usize, decodes as u32);
        let (mut prefills, mut decodes) = lists(prefills, decodes);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let decision = decide_over(&spec, &facts(), &prefills, &decodes, &classes);
        // A pair's first decision over no candidate is emptied by no
        // stage.
        let Decision::Shed(shed) = &decision else {
            panic!("not shed: {decision:?}");
        };
        assert_eq!((shed.emptied_by, shed.side), ("none", Some(side)));
        assert!(!prefills_full(&decision));
    }
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

#[test]
fn a_pairs_issue_tells_each_side_once_under_its_own_lease() {
    let pair = decided(1);
    let facts = keyed();
    let issued = issue(
        pair.spec.engine(),
        &facts,
        &pair.decision,
        LeaseId(9),
        false,
    );
    let (lease, session) = issued.expect("a pair's lease");
    assert_eq!(
        (lease.id, lease.side, lease.booked, lease.target_index),
        (LeaseId(9), Side::Prefill, WorkerId(1), 0)
    );
    assert_eq!(
        (lease.generation, lease.tokens, lease.issued_micros),
        (3, 40, 7_000)
    );
    assert_eq!(lease.session, facts.session());
    assert_ne!(session, lease.session, "each side keeps its own session");
    assert_eq!(
        pair.words(),
        [
            ("issued", Side::Prefill, 1, 0, lease.session, 7_000, 1),
            ("issued", Side::Decode, 101, 0, session, 7_000, 1),
        ]
    );
}

#[test]
fn a_decode_sent_later_is_not_told_with_its_prefill() {
    let pair = decided(1);
    let issued = issue(
        pair.spec.engine(),
        &keyed(),
        &pair.decision,
        LeaseId(9),
        true,
    );
    let (lease, _) = issued.expect("a pair's lease");
    assert_eq!(
        pair.words(),
        [("issued", Side::Prefill, 1, 0, lease.session, 7_000, 1)]
    );
}

#[test]
fn a_decision_that_picks_no_pair_issues_nothing() {
    let pair = decided(1);
    let none = shed(ShedReason::NoEndpoints, WIRES, 0, 3);
    assert!(issue(pair.spec.engine(), &facts(), &none, LeaseId(1), false).is_none());
    assert!(pair.words().is_empty());
}

/// The second ask runs over the records of its own instant: the held
/// prefill's partners, with the room each has then. Its pick names the
/// held prefill, and its issue is told dated at the call under the
/// lease's own id.
#[test]
fn the_decode_asked_again_is_decided_over_the_held_prefills_partners() {
    let pair = decided(2);
    let facts = keyed();
    let issued = issue(pair.spec.engine(), &facts, &pair.decision, LeaseId(9), true);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    // At the call the first decode has no room and the second has one.
    let (mut prefills, mut decodes) = lists(1, 2);
    let classes = every(1, 2);
    stamp(
        &mut prefills,
        &mut decodes,
        &classes,
        &Gates(|_| true, |at| at == 1),
    );
    let again = decide_decode_records(
        pair.spec.engine(),
        &facts,
        (&Constraints::default(), unvetoed(None)),
        &pool(&decodes),
        pools(&prefills, &decodes, &classes),
        &prefill_alone(WorkerId(1)),
        &mut Scratch::default(),
    );
    assert_eq!(picked(&again), (1, 102));
    let decode = issue_decode(pair.spec.engine(), &facts, &again, &lease, 9_500);
    let decode = decode.expect("the decode side's lease");
    assert_eq!(
        (decode.id, decode.side, decode.booked, decode.session),
        (LeaseId(9), Side::Decode, WorkerId(102), session)
    );
    assert_eq!(
        decode.issued_micros, 7_000,
        "the lease keeps its first instant"
    );
    assert_eq!(
        pair.words(),
        [("issued", Side::Decode, 102, 0, session, 9_500, 1)]
    );
}

#[test]
fn a_prefill_that_left_its_list_sheds_the_second_ask() {
    let pair = decided(1);
    let (mut prefills, mut decodes) = lists(1, 1);
    let classes = every(1, 1);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let again = decide_decode_records(
        pair.spec.engine(),
        &facts(),
        (&Constraints::default(), unvetoed(None)),
        &pool(&decodes),
        pools(&prefills, &decodes, &classes),
        &prefill_alone(WorkerId(5)),
        &mut Scratch::default(),
    );
    match again {
        Decision::Shed(shed) => assert_eq!(shed.emptied_by, "prefill-gone"),
        other => panic!("not shed: {other:?}"),
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
            facts: RequestFacts { seed, ..facts() },
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

/// A placement says a pair's words as a guard driven by hand says
/// them, each side's under its own lease and the index of the target
/// taken. It writes into each end the worker that served, from that
/// target, whatever its host wrote there. Its two workers are the
/// lists' own objects at the target's places. An encode side's send
/// and end say nothing. A pair told such an end behind its two sends
/// still owes both sides. A pair whose host said those two alone and
/// let go is released whole, as a guard dropped unsent.
#[test]
fn a_placement_says_a_pairs_words_and_names_the_workers_that_served() {
    let host: &'static Host = Box::leak(Box::new(Host::of(1, 2)));
    let classes = every(1, 2);
    let asks = host.asks(&classes, false);
    let whole = || {
        let now = Instant::now();
        let answer = host.engine.decide_under(GatewayId::LOCAL, &asks, now);
        let (decided, later) = answer
            .expect("an answer")
            .whole()
            .expect("a whole decision");
        let taken = target_at(&decided.decision, 1);
        (decided, later, now, taken, taken.pair.expect("a pair"))
    };
    // What the stages heard since the last call: each word with its
    // side and its lease's index, and the worker each completion names.
    let words = || {
        let heard = std::mem::take(&mut *host.heard.lock());
        let told: Vec<_> = heard.iter().map(|word| (word.0, word.1, word.3)).collect();
        let ended = heard.iter().filter(|word| word.0 == "completed");
        let served: Vec<_> = ended.map(|word| (word.1, WorkerId(word.6))).collect();
        (told, served)
    };

    let (decided, later, now, taken, pair) = whole();
    let profile = decided.decision.pick().expect("a pick").trace.profile;
    assert_ne!(profile, 0, "the document's second profile picked");
    let prefill = &host.prefills[taken.position as usize];
    let decode = &host.decodes[pair.decode_position as usize];
    let placement = crate::tracking::Placement::taken(
        host.engine.loaded(),
        decided,
        later,
        now,
        1,
        Arc::clone(prefill),
        Some(Arc::clone(decode)),
    );
    let mut placement = placement.expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, prefill));
    assert!(Arc::ptr_eq(
        placement.decode.as_ref().expect("a decode"),
        decode
    ));
    let ids = (placement.id(), placement.decode_id());
    assert_eq!(ids, (taken.id, Some(pair.decode_id)));
    assert_eq!(placement.profile, "prefill");
    placement.sent(Side::Prefill);
    placement.sent(Side::Decode);
    placement.ended(Side::Encode, outcome(Side::Encode, WorkerId(77)));
    // Its host names no worker and another side in each outcome.
    placement.ended(Side::Prefill, outcome(Side::Single, WorkerId(0)));
    assert!(placement.owes_events(), "the decode side has not ended");
    placement.ended(Side::Decode, outcome(Side::Single, WorkerId(0)));
    assert!(!placement.owes_events());
    drop(placement);
    let (decided, later, now, _, _) = whole();
    let unsent = crate::tracking::Placement::taken(
        host.engine.loaded(),
        decided,
        later,
        now,
        1,
        Arc::clone(prefill),
        Some(Arc::clone(decode)),
    );
    let mut unsent = unsent.expect("a placement");
    unsent.sent(Side::Encode);
    unsent.ended(Side::Encode, outcome(Side::Encode, WorkerId(77)));
    drop(unsent);
    let (through_placements, served) = words();
    let named = [(Side::Prefill, taken.id), (Side::Decode, pair.decode_id)];
    assert_eq!(served, named);

    let (decided, later, now, _, _) = whole();
    let mut guard = DecisionGuard::holding(&host.engine, GatewayId::LOCAL, decided, later, now);
    let taken = guard.take(1).expect("a target");
    let decode = taken.pair.expect("a pair").decode_id;
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, taken.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    drop(guard);
    let (decided, later, now, _, _) = whole();
    let mut guard = DecisionGuard::holding(&host.engine, GatewayId::LOCAL, decided, later, now);
    assert!(guard.take(1).is_some());
    drop(guard);
    assert_eq!(through_placements, words().0);
    let told: Vec<_> = through_placements.iter().map(|word| word.0).collect();
    // Each word is told of both sides: the pair sent and ended, then
    // the pair let go.
    let sent_and_ended = ["issued", "dispatched", "completed"].map(|word| [word; 2]);
    let let_go = ["issued", "released"].map(|word| [word; 2]);
    assert_eq!(told, [sent_and_ended.concat(), let_go.concat()].concat());
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 0));
}

/// A decision names each worker by its place in the list its host
/// handed and by its id, and reads the workers and the host's admission
/// as they stand at the call.
#[test]
fn a_pair_is_decided_over_the_hosts_two_lists_as_they_stand_at_the_call() {
    let host = Host::of(2, 3);
    let mut seen = [false; 2];
    for seed in 0..16 {
        seen[host.picked(&host.decide(&OPEN, seed, None)).0] = true;
    }
    assert_eq!(seen, [true; 2], "the rotation reaches both prefills");
    // A prefill that stopped being ready and a decode with no room: the
    // next decision reads both where the host keeps them.
    host.prefills[0].set_status(WorkerStatus::NotReady);
    let one_room = Gates(|_| true, |at| at == 2);
    assert_eq!(host.picked(&host.decide(&one_room, 1, None)), (1, 2));
    // The one prefill left is at its limit: the request waits.
    let full = Gates(|_| false, |_| true);
    assert!(prefills_full(&host.decide(&full, 1, None)));
}

/// A pair its host holds to places is decided over the pools' own lists,
/// and each side opens with its own places: no prefill and no decode
/// outside them is picked, under the stock document and under one whose
/// profiles list no filter. One side may be held alone. A side with no
/// place kept picks no pair. The decode asked again is held to the
/// decodes' places of its own call.
#[test]
fn a_pair_held_to_places_is_handed_no_other_place_on_either_side() {
    for host in [Host::of(3, 4), Host::unfiltered(3, 4)] {
        // Both sides held: every pick is of the places, and the turns
        // reach each of them.
        let mut seen = ([false; 3], [false; 4]);
        for seed in 0..64 {
            let both = [Some(&[1][..]), Some(&[0, 3][..])];
            let picked =
                host.picked(&host.decide_among(&OPEN, seed, None, Constraints::default(), both));
            seen.0[picked.0] = true;
            seen.1[picked.1] = true;
        }
        assert_eq!(seen, ([false, true, false], [true, false, false, true]));
        for seed in 0..32 {
            let prefills = [Some(&[0, 2][..]), None];
            let picked = host.picked(&host.decide_among(
                &OPEN,
                seed,
                None,
                Constraints::default(),
                prefills,
            ));
            assert_ne!(picked.0, 1, "seed {seed}");
            let decodes = [None, Some(&[2][..])];
            let picked =
                host.picked(&host.decide_among(&OPEN, seed, None, Constraints::default(), decodes));
            assert_eq!(picked.1, 2, "seed {seed}");
        }
        // No decode kept: no pair is picked, whatever is ready.
        let none = [None, Some(&[][..])];
        assert!(host
            .decide_among(&OPEN, 1, None, Constraints::default(), none)
            .pick()
            .is_none());
        // The decode asked again behind a sent prefill.
        let (sent, _) = host.picked(&host.decide(&OPEN, 3, None));
        let held = prefill_alone(worker_id(host.prefills[sent].as_ref()));
        for seed in 0..16 {
            let decodes = [None, Some(&[1][..])];
            let again =
                host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), decodes);
            assert_eq!(host.picked(&again), (sent, 1), "seed {seed}");
        }
    }
}

/// A decode among the decode places is handed only with the prefill it
/// partners. Two prefills, each with its own decode, both decode places
/// stated and the prefill list held to the second: the pair is the
/// second prefill with the second decode at every turn. So it is where
/// a place of either side stands past its list's end, which names no
/// row. With the prefill list free and the first decode not ready, the
/// second decode place alone opens a prefill: the same pair. The
/// gateway's classic router picks a pair's decode among the picked
/// prefill's own partners that are open for the request
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1389-1394`, `:1548-1553`, at main `b8c15d40a`).
#[test]
fn a_decode_among_the_places_is_handed_only_with_the_prefill_it_partners() {
    let (spec, _) = hearing();
    let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
    let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
    let free = Constraints::default();
    let decide = |seed: u64, first_ready: bool, places: [Option<&[u32]>; 2]| {
        let (mut prefills, mut decodes) = lists(2, 2);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        decodes[0].health.healthy = first_ready;
        let facts = RequestFacts { seed, ..facts() };
        decide_records(
            spec.engine(),
            &facts,
            (&free, places.map(unvetoed)),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        )
    };
    for seed in 0..24 {
        let held = [Some(&[1][..]), Some(&[0, 1][..])];
        assert_eq!(picked(&decide(seed, true, held)), (2, 102), "seed {seed}");
        let past = [Some(&[1, 7][..]), Some(&[0, 1, 9][..])];
        assert_eq!(picked(&decide(seed, true, past)), (2, 102), "seed {seed}");
        let second = [None, Some(&[0, 1][..])];
        assert_eq!(
            picked(&decide(seed, false, second)),
            (2, 102),
            "seed {seed}"
        );
    }
}

/// The decode asked again is held to the decode places that partner its
/// held prefill. Two prefills, each with its own decode, and the second
/// prefill held, which is not the first of its list: among both decode
/// places, and at the second alone, its decode is the second at every
/// turn. At the first place alone the places and the prefill's partners
/// meet in no decode, and the call is shed under the partners' name. So
/// it is over no place at all. The gateway's classic router asks no
/// decode again: it selects a request's pair whole, once, the decode
/// among the picked prefill's own partners
/// (`model_gateway/src/routers/common/placement.rs:1227-1275`,
/// `:1548-1553`, at main `b8c15d40a`).
#[test]
fn the_decode_asked_again_is_held_to_the_places_that_partner_its_prefill() {
    let pair = decided(2);
    let (mut prefills, mut decodes) = lists(2, 2);
    let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
    let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let free = Constraints::default();
    let again = |seed: u64, places: &[u32]| {
        let facts = RequestFacts { seed, ..facts() };
        decide_decode_records(
            pair.spec.engine(),
            &facts,
            (&free, unvetoed(Some(places))),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            &prefill_alone(WorkerId(2)),
            &mut Scratch::default(),
        )
    };
    for seed in 0..24 {
        assert_eq!(picked(&again(seed, &[0, 1])), (2, 102), "seed {seed}");
        assert_eq!(picked(&again(seed, &[1])), (2, 102), "seed {seed}");
    }
    let nobody = (ShedReason::NoEndpoints, "partners");
    assert_eq!(shed_for(&again(1, &[0])), nobody);
    assert_eq!(shed_for(&again(1, &[])), nobody);
}

/// A side's places stand beside what the request states, and hold the
/// other side to nothing. Three prefills and four decodes, every pair
/// open. With the prefill list held to its second place the turns reach
/// every decode, and with the decode list held to its fourth they reach
/// every prefill. A decode its host just failed to reach steps aside
/// among the decode places while another of them is left. A prefill
/// list held to no place picks no pair, as a pair with no prefill picks
/// none. Behind the third prefill the
/// decode asked again reads the decodes' places alone: the prefills'
/// places, which leave that prefill out, are not read, and decode
/// places that keep none pick none. The gateway's classic router steps
/// aside from the engines a request tried on each side of a pair, among
/// the workers its request's rule admits
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1492-1494`, `:1568-1570`, at main `b8c15d40a`).
#[test]
fn a_sides_places_stand_beside_the_requests_word_and_hold_the_other_side_to_nothing() {
    let host = Host::of(3, 4);
    let free = Constraints::default;
    let mut decodes = [false; 4];
    for seed in 0..64 {
        let prefill = [Some(&[1][..]), None];
        let picked = host.picked(&host.decide_among(&OPEN, seed, None, free(), prefill));
        assert_eq!(picked.0, 1, "seed {seed}");
        decodes[picked.1] = true;
    }
    assert_eq!(decodes, [true; 4]);
    let mut prefills = [false; 3];
    for seed in 0..64 {
        let decode = [None, Some(&[3][..])];
        let picked = host.picked(&host.decide_among(&OPEN, seed, None, free(), decode));
        assert_eq!(picked.1, 3, "seed {seed}");
        prefills[picked.0] = true;
    }
    assert_eq!(prefills, [true; 3]);
    let tried = [worker_id(host.decodes[0].as_ref())];
    for seed in 0..16 {
        let excluding = Constraints {
            excluded: &tried,
            ..Constraints::default()
        };
        let among = [None, Some(&[0, 3][..])];
        let picked = host.picked(&host.decide_among(&OPEN, seed, None, excluding, among));
        assert_eq!(picked.1, 3, "seed {seed}");
    }
    let none = [Some(&[][..]), None];
    let unheld = host.decide_among(&OPEN, 1, None, free(), none);
    assert_eq!(shed_for(&unheld), (ShedReason::NoEndpoints, "none"));
    let held = prefill_alone(worker_id(host.prefills[2].as_ref()));
    for seed in 0..16 {
        let other = [Some(&[0][..]), Some(&[1][..])];
        let again = host.decide_among(&OPEN, seed, Some(&held), free(), other);
        assert_eq!(host.picked(&again), (2, 1), "seed {seed}");
    }
    let none = [None, Some(&[][..])];
    let again = host.decide_among(&OPEN, 1, Some(&held), free(), none);
    assert_eq!(shed_for(&again), (ShedReason::NoEndpoints, "partners"));
}

/// The decode asked again keeps the decode its first decision named
/// while that decode stands among the decodes' places of the call, and
/// picks among the places where it does not: no decode outside them is
/// handed, kept or not. One prefill and three decodes under round
/// robin, which keeps its pick. The gateway's classic router asks no
/// decode again: it selects a request's pair whole, once
/// (`model_gateway/src/routers/common/placement.rs:1227-1275`, at main
/// `b8c15d40a`).
#[test]
fn the_decode_asked_again_keeps_its_first_decode_among_the_places() {
    let host = Host::under(presets("round_robin"), 1, 3);
    for seed in 0..12 {
        let held = first_target(&host.decide(&OPEN, seed, None));
        let named = held.pair.expect("a decode").decode_position;
        let other = (named + 1) % 3;
        // A host states its places ascending.
        let (with, without) = ([named.min(other), named.max(other)], [other]);
        let among = [None, Some(&with[..])];
        let again = host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), among);
        assert_eq!(host.picked(&again).1, named as usize, "seed {seed}");
        let outside = [None, Some(&without[..])];
        let again = host.decide_among(&OPEN, seed, Some(&held), Constraints::default(), outside);
        assert_eq!(host.picked(&again).1, other as usize, "seed {seed}");
    }
}

/// A pair's first pick states what its prefill holds of the request
/// ([`Pick::overlap`](crate::engine::Pick::overlap)), where the
/// request's own profile reads who holds the prompt at the prefill
/// phase: with no prompt sent before, a miss of the request's blocks.
/// The decode asked again states nothing: its prefill ran, and no
/// column of the prefills is read for it. Neither does a pair whose
/// prefill side reads no holder, whatever its decode side reads.
#[test]
fn a_pairs_first_pick_states_what_its_prefill_holds() {
    use crate::engine::decision::Overlap;

    let under = |prefill: &str, decode: &str| {
        let document = PolicySpec::pair_presets(prefill, decode).expect("the pair presets");
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        Host::under((spec, Arc::default()), 2, 2)
    };
    let stated = |decision: &Decision| decision.pick().expect("a pick").overlap();

    let host = under("cache_aware", "cache_aware");
    let first = host.decide(&OPEN, 3, None);
    let miss = stated(&first);
    assert_ne!(miss.of, 0, "a miss is stated of the request's blocks");
    let nothing_held = Overlap {
        of: miss.of,
        ..Overlap::NONE
    };
    assert_eq!(miss, nothing_held);
    let held = first_target(&first);
    let again = host.decide(&OPEN, 4, Some(held.id));
    assert_eq!(stated(&again), Overlap::NONE, "the decode asked again");

    let decode_alone = under("round_robin", "cache_aware");
    let first = decode_alone.decide(&OPEN, 3, None);
    assert_eq!(
        stated(&first),
        Overlap::NONE,
        "a prefill side that reads none"
    );
}

/// The decode asked again reads both lists at its own call, and keeps
/// the prefill that was sent.
#[test]
fn the_decode_asked_again_reads_the_hosts_lists_of_its_own_instant() {
    let host = Host::of(2, 2);
    let (sent, chosen) = host.picked(&host.decide(&OPEN, 3, None));
    let held = worker_id(host.prefills[sent].as_ref());
    // The decode the first pick named stops being ready behind it.
    host.decodes[chosen].set_status(WorkerStatus::NotReady);
    let again = host.decide(&OPEN, 4, Some(held));
    assert_eq!(host.picked(&again), (sent, 1 - chosen));
    // A prefill no list holds any more sheds the call.
    match host.decide(&OPEN, 5, Some(WorkerId(held.0 ^ 1))) {
        Decision::Shed(shed) => assert_eq!(shed.emptied_by, "prefill-gone"),
        other => panic!("not shed: {other:?}"),
    }
}

/// The decode asked again reads one prefill, the held one: its host's
/// prefill limit is asked of that prefill's place alone.
#[test]
fn the_decode_asked_again_asks_the_limit_of_its_held_prefill_alone() {
    /// A host with no limit that keeps the places its prefill limit
    /// was asked of.
    #[derive(Debug, Default)]
    struct Asked(RefCell<Vec<usize>>);

    impl PairAdmission for Asked {
        fn has_capacity(&self, prefill: usize) -> bool {
            self.0.borrow_mut().push(prefill);
            true
        }

        fn has_room(&self, _: usize) -> bool {
            true
        }
    }

    let (host, asked) = (Host::of(3, 2), Asked::default());
    let held = worker_id(host.prefills[1].as_ref());
    let again = host.decide(&asked, 4, Some(held));
    assert_eq!(host.picked(&again).0, 1);
    assert_eq!(asked.0.take(), [1]);
    // The thread's buffer holds the one prefill record written.
    let written = BUFFERS.with(|buffers| buffers.borrow().rows[0].len());
    assert_eq!(written, 1);
}

/// A decode asked again where its host lists no decode is refused for
/// the partners it lacks: the held prefill stands in its list. So it is
/// behind a target that names the decode of its first decision. A
/// prefill that left is still named where a list holds no record.
///
/// The classic router places both sides of a pair in one call
/// (`placement.rs:1227`) and decides no decode again, so this answer
/// is the engine's own. Each refusal names its side: the decodes' for
/// the partners the call lacks, and the prefills' for the held prefill
/// that left its list.
#[test]
fn a_decode_asked_again_over_no_decode_is_not_shed_as_a_prefill_that_left() {
    let shed_under = |host: &Host, held: &Target| match host.decide_held(&OPEN, 4, Some(held)) {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by, shed.side),
        other => panic!("not shed: {other:?}"),
    };
    let host = Host::of(1, 0);
    let held = worker_id(host.prefills[0].as_ref());
    let no_partner = (ShedReason::NoEndpoints, "partners", Some(PairSide::Decode));
    assert_eq!(shed_under(&host, &prefill_alone(held)), no_partner);
    let first = first_target(&Host::of(1, 1).decide(&OPEN, 3, None));
    assert_eq!((first.id, first.pair.is_some()), (held, true));
    assert_eq!(shed_under(&host, &first), no_partner);
    let gone = (
        ShedReason::NoEndpoints,
        "prefill-gone",
        Some(PairSide::Prefill),
    );
    let left = prefill_alone(WorkerId(held.0 ^ 1));
    assert_eq!(shed_under(&host, &left), gone);
    assert_eq!(shed_under(&Host::of(0, 1), &prefill_alone(held)), gone);
}

/// Over no decode, as over two lists with records, a request that
/// states a demand is shed under the demand's stage. Over no prefill
/// the call is shed for the prefill that left, before a demand is read.
/// The classic router decides no decode again (`placement.rs:1227`):
/// this answer is the engine's own too.
#[test]
fn a_decode_asked_again_over_no_decode_sheds_a_stated_demand_under_its_stage() {
    let pair = decided(1);
    let demanded = Constraints {
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..Constraints::default()
    };
    let shed_under = |prefills: u8, decodes: u8| {
        let classes = every(usize::from(prefills), u32::from(decodes));
        let (mut prefills, mut decodes) = lists(u64::from(prefills), u64::from(decodes));
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        let again = decide_decode_records(
            pair.spec.engine(),
            &facts(),
            (&demanded, unvetoed(None)),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            &prefill_alone(WorkerId(1)),
            &mut Scratch::default(),
        );
        match again {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed: {other:?}"),
        }
    };
    assert_eq!(shed_under(1, 0), (ShedReason::NoEndpoints, WIRES));
    assert_eq!(shed_under(0, 1), (ShedReason::NoEndpoints, "prefill-gone"));
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

/// A pair that states a hard demand is decided over its two lists,
/// each side held to the workers that meet the demand: every pick
/// names one of them on each side, and each of them takes its turn.
/// With no demand the other workers take theirs.
///
/// The classic router holds both of a pair's pools to the request's
/// own rule (`model_gateway/src/routers/common/placement.rs:1227`, the
/// rule's read at `:1262-1266`; its cases at `:3272` and `:3324`, at
/// main `b8c15d40a`).
#[test]
fn a_pair_that_states_a_demand_is_decided_among_the_workers_of_each_side_that_meet_it() {
    // The second prefill takes images, and so do the first and the
    // third decode.
    let host = stating(2, 3, |side, at| FixedFacts {
        vision: if side == 0 { at == 1 } else { at != 1 },
        ..FixedFacts::default()
    });
    let pixels = demanding(Demands {
        vision: true,
        ..Demands::default()
    });
    let seeing = (BTreeSet::from([1]), BTreeSet::from([0, 2]));
    assert_eq!(places_named(&host, pixels, [None, None]), seeing);
    let all = (BTreeSet::from([0, 1]), BTreeSet::from([0, 1, 2]));
    assert_eq!(
        places_named(&host, Constraints::default(), [None, None]),
        all
    );
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

/// A demand is judged among the places a pair's host states for a
/// side: the side is held to those of them whose worker meets it, and
/// a side none of whose places meets it sheds the pair, whatever a
/// worker outside them states.
#[test]
fn a_pairs_demand_is_judged_among_the_places_its_host_states() {
    // The second and the third worker of each list take images.
    let host = stating(3, 3, |_, at| FixedFacts {
        vision: at != 0,
        ..FixedFacts::default()
    });
    let pixels = demanding(Demands {
        vision: true,
        ..Demands::default()
    });
    let held = places_named(&host, pixels, [Some(&[0, 1]), Some(&[1, 2])]);
    assert_eq!(held, (BTreeSet::from([1]), BTreeSet::from([1, 2])));
    let blind = host.decide_among(&OPEN, 0, None, pixels, [None, Some(&[0])]);
    assert_eq!(shed_of(&blind), (ShedReason::NoEndpoints, "vision"));
}

/// A pair's request that states both a hard demand and places of its
/// own is declined under the places' name, whether its sides meet the
/// demand or not: a pair's places are its lists' to state, and that
/// answer comes ahead of the demand's.
#[test]
fn a_pair_that_states_a_demand_and_places_of_its_own_is_declined_under_the_places() {
    let pixels = demanding(Demands {
        vision: true,
        ..Demands::default()
    });
    let stated = Constraints {
        places: Some(&[0]),
        ..pixels
    };
    let declined = (ShedReason::PolicyDeclined, single::PLACES);
    // Every worker takes images, then none does.
    for sees in [true, false] {
        let host = stating(1, 1, |_, _| FixedFacts {
            vision: sees,
            ..FixedFacts::default()
        });
        let decision = host.decide_among(&OPEN, 0, None, stated, [None, None]);
        assert_eq!(shed_of(&decision), declined, "takes images: {sees}");
    }
}

/// The decode asked again behind a held prefill is held to the decodes
/// that meet the demand, and judges no prefill: the held one ran
/// already, whatever it states. Where no decode meets the demand the
/// call is shed under that demand's stage.
///
/// The classic router decides no decode again (`placement.rs:1227`, at
/// main `b8c15d40a`): this answer is the engine's own.
#[test]
fn a_decode_asked_again_is_held_to_the_decodes_that_meet_the_demand() {
    let pixels = demanding(Demands {
        vision: true,
        ..Demands::default()
    });
    // No prefill takes images, and the second decode alone does.
    let host = stating(1, 2, |side, at| FixedFacts {
        vision: (side, at) == (1, 1),
        ..FixedFacts::default()
    });
    let held = prefill_alone(worker_id(host.prefills[0].as_ref()));
    for seed in 0..6 {
        let again = host.decide_among(&OPEN, seed, Some(&held), pixels, [None, None]);
        assert_eq!(host.picked(&again), (0, 1));
    }
    // The pair's first decision judges its prefills.
    let first = host.decide_among(&OPEN, 0, None, pixels, [None, None]);
    assert_eq!(shed_of(&first), (ShedReason::NoEndpoints, "vision"));
    assert_eq!(side_of(&first), Some(PairSide::Prefill));
    // No decode takes images.
    let blind = stating(1, 2, |_, _| FixedFacts::default());
    let held = prefill_alone(worker_id(blind.prefills[0].as_ref()));
    let again = blind.decide_among(&OPEN, 0, Some(&held), pixels, [None, None]);
    assert_eq!(shed_of(&again), (ShedReason::NoEndpoints, "vision"));
    assert_eq!(
        side_of(&again),
        Some(PairSide::Decode),
        "no prefill is judged"
    );
}

/// Under round robin on both sides a pair's first decision takes the
/// decode rotation's turn, and the decode decided again is handed the
/// decode that decision named and keeps it. So requests that come one
/// at a time reach both decodes evenly, and each second answer is its
/// first. Handed the prefill alone, every second answer was the other
/// decode. A first decode that stopped being ready is left for another.
#[test]
fn the_decode_asked_again_keeps_its_first_decode_under_round_robin() {
    let host = Host::under(presets("round_robin"), 1, 2);
    let mut sent = [0; 2];
    for seed in 0..8 {
        let held = first_target(&host.decide(&OPEN, seed, None));
        let named = held.pair.expect("a decode").decode_position as usize;
        let again = host.decide_held(&OPEN, seed, Some(&held));
        assert_eq!(host.picked(&again).1, named, "seed {seed}");
        sent[named] += 1;
    }
    assert_eq!(sent, [4; 2]);
    let held = first_target(&host.decide(&OPEN, 8, None));
    let named = held.pair.expect("a decode").decode_position as usize;
    let alone = host.decide(&OPEN, 8, Some(held.id));
    assert_eq!(host.picked(&alone).1, 1 - named, "no decode handed");
    let held = first_target(&host.decide(&OPEN, 9, None));
    let named = held.pair.expect("a decode").decode_position as usize;
    host.decodes[named].set_status(WorkerStatus::NotReady);
    let again = host.decide_held(&OPEN, 9, Some(&held));
    assert_eq!(host.picked(&again).1, 1 - named);
}

/// The decode asked again through the guard is handed the decode of the
/// target the guard holds. Under round robin each second answer is its
/// first, and requests one at a time reach both decodes evenly.
#[test]
fn the_guards_decode_call_keeps_its_first_decode() {
    let host = Host::under(presets("round_robin"), 1, 2);
    let classes = every(1, 2);
    let mut sent = [0; 2];
    for _ in 0..4 {
        let (asks, now) = (host.asks(&classes, true), Instant::now());
        let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
        let mut guard = guard.expect("a decision");
        let named = host.picked(guard.decision()).1;
        guard.dispatched(now);
        let again = *guard.decide_decode_under(&asks, now).expect("a decision");
        assert!(!guard.decode_chosen(), "the call picked");
        assert_eq!(host.picked(&again).1, named);
        sent[named] += 1;
    }
    assert_eq!(sent, [2; 2]);
}

/// A host that takes another target of its pick holds that target, and
/// the guard hands it to the decode asked again. Under round robin the
/// answer's first decode is then the taken target's: for a target taken
/// before its prefill went out, and for a target's decode taken behind
/// the prefill. The pick's third target is taken: its decode is neither
/// the first target's nor the one the rotation's next turn names.
#[test]
fn the_guards_decode_call_keeps_the_decode_its_host_took() {
    let host = Host::under(presets("round_robin"), 1, 3);
    let classes = every(1, 3);
    for behind_the_prefill in [false, true] {
        let (asks, now) = (host.asks(&classes, true), Instant::now());
        let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
        let mut guard = guard.expect("a decision");
        let first = host.picked(guard.decision()).1;
        let taken = if behind_the_prefill {
            guard.dispatched(now);
            guard.take_decode(2)
        } else {
            let taken = guard.take(2);
            guard.dispatched(now);
            taken
        };
        let taken = taken.and_then(|target| target.pair);
        let taken = taken.expect("a third target").decode_position as usize;
        assert_ne!(taken, first, "another decode than the first target's");
        let again = *guard.decide_decode_under(&asks, now).expect("a decision");
        assert_eq!(host.picked(&again).1, taken, "{behind_the_prefill}");
    }
}

/// A decode that is one rank of a data-parallel engine is kept by its
/// worker and its rank. Over an engine's two ranks each second answer is
/// the rank its first named, and both ranks are named.
#[test]
fn a_ranked_decode_is_kept_by_its_rank() {
    let rank = |rank: usize| -> Arc<dyn Worker> {
        let spec = WorkerSpec {
            worker_type: WorkerType::Decode,
            ..WorkerSpec::new("grpc://decode:9000".to_string())
        };
        let worker = <BasicWorkerBuilder>::from_spec(spec)
            .dp_config(rank, 2)
            .health_config(no_health_check())
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    };
    let prefill = worker("grpc://prefill:9000".to_string(), WorkerType::Prefill);
    let ranks = Arc::from(vec![rank(0), rank(1)]);
    let host = Host::over(presets("round_robin"), Arc::from(vec![prefill]), ranks);
    let mut named_ranks = [0; 2];
    for seed in 0..4 {
        let held = first_target(&host.decide(&OPEN, seed, None));
        let named = held.pair.expect("a decode");
        let again = first_target(&host.decide_held(&OPEN, seed, Some(&held)));
        assert_eq!(again.pair, Some(named), "seed {seed}");
        named_ranks[usize::from(named.decode_dp_rank)] += 1;
    }
    assert_eq!(named_ranks, [2; 2], "both ranks are named");
}

/// The presets' own filters stand ahead of a kept decode. A first decode
/// the request excludes is left for another decode while one is kept,
/// and is handed again where it is the only decode.
#[test]
fn an_excluded_first_decode_is_kept_only_where_it_is_alone() {
    for (decodes, kept) in [(2, false), (1, true)] {
        let host = Host::under(presets("round_robin"), 1, decodes);
        let held = first_target(&host.decide(&OPEN, 0, None));
        let first = held.pair.expect("a decode");
        let excluded = [first.decode_id];
        let again = host.decide_excluding(&OPEN, 0, Some(&held), &excluded);
        let named = first.decode_position as usize;
        assert_eq!(host.picked(&again).1 == named, kept, "{decodes}");
    }
}

/// `passthrough` takes no turn, so its decode asked again is decided
/// again, whatever the call is handed. A pair's first decision names
/// the first decode of the list that has a room. A decode ahead of that
/// one has its room by the second call, and the answer is that decode.
#[test]
fn the_decode_asked_again_under_passthrough_is_the_lists_first() {
    let host = Host::under(presets("passthrough"), 1, 3);
    let no_room_ahead = Gates(|_| true, |at| at != 0);
    let held = first_target(&host.decide(&no_room_ahead, 0, None));
    assert_eq!(held.pair.expect("a decode").decode_position, 1);
    let again = host.decide_held(&OPEN, 0, Some(&held));
    assert_eq!(host.picked(&again).1, 0);
    let alone = host.decide(&OPEN, 0, Some(held.id));
    assert_eq!(host.picked(&alone).1, 0, "no decode handed");
}

/// Both sides of a pair take the hash ring their caller lends, each
/// over its own list. The gateway hands a pair its model's one ring
/// where either side's policy reads it
/// (`model_gateway/src/routers/common/placement.rs`, lines 1574 to 1585
/// at 882be6f4f), and each side's policy places by its two rules
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 99 to
/// 140). A side takes the ring's worker only where its own list holds
/// an eligible one: the walk passes the other side's workers, and a
/// worker of its own that is not ready. A value no worker of the side
/// takes goes by its position over that side's eligible workers.
#[test]
fn each_side_of_a_pair_places_on_the_ring_its_caller_lends() {
    let document = PolicySpec::pair_presets("consistent_hashing", "consistent_hashing");
    let spec = AppliedSpec::validate(&document.expect("the pair presets"));
    let host = Host::under((spec.expect("a valid document"), Arc::default()), 3, 3);
    let lent = |workers: &str| {
        let address = |at: &str| format!("grpc://{at}:9000");
        InOrder(workers.split(' ').map(address).collect())
    };
    // The pair a keyed request is placed on where it stands at
    // `position` of the ring lent.
    let placed = |ring: &InOrder, position: u64| {
        let [prefill, decode] = &host.lists;
        let classes = every(3, 3);
        let lists = PairLists {
            prefill: prefill.lent(),
            decode: decode.lent(),
            classes: &classes,
            admission: &OPEN,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        };
        let positions = [position];
        let facts = RequestFacts {
            ring: Some(ring),
            ring_positions: &positions,
            ..keyed()
        };
        let mut request = Request {
            id: b"",
            facts,
            constraints: Constraints::default(),
        };
        let now = Instant::now();
        host.picked(&lists.decide(&host.engine, None, now, &mut request, None))
    };
    // The model's one ring holds both lists' workers: each side takes
    // the first worker of its own list that the walk reaches.
    let model = lent("decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2");
    assert_eq!(placed(&model, 0), (2, 1));
    assert_eq!(placed(&model, 3), (0, 2));
    // A worker that is not ready is passed over on its side.
    host.prefills[2].set_status(WorkerStatus::NotReady);
    host.decodes[1].set_status(WorkerStatus::NotReady);
    assert_eq!(placed(&model, 0), (0, 0));
    // A ring that reaches no eligible worker of a side: the value goes
    // by its position over that side's eligible workers, two here.
    let no_prefill = lent("decode-2 decode-0 prefill-2");
    assert_eq!(placed(&no_prefill, 3), (1, 2));
    let no_decode = lent("prefill-1 decode-1");
    assert_eq!(placed(&no_decode, 4), (1, 0));
}

/// Under the prefix form each side of a pair is placed on the lent ring
/// too, from the first place its caller states whose owner is within
/// the bound: the first eligible worker of its own list that the walk
/// reaches there, as the gateway's own policy takes it from the pair's
/// one ring (`model_gateway/src/policies/prefix_hash.rs`, lines 291 to
/// 315 at 882be6f4f). Every worker here is idle: the first place is
/// the one taken.
#[test]
fn each_side_of_a_pair_under_the_prefix_form_places_on_the_lent_ring() {
    let document = PolicySpec::pair_presets("prefix_hash", "prefix_hash");
    let spec = AppliedSpec::validate(&document.expect("the pair presets"));
    let host = Host::under((spec.expect("a valid document"), Arc::default()), 3, 3);
    let address = |at: &str| format!("grpc://{at}:9000");
    let model = "decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2";
    let ring = InOrder(model.split(' ').map(address).collect());
    let placed = |position: u64| {
        let classes = every(3, 3);
        let positions = [position];
        // The host's own request over its two lists, every pair open.
        let Asks(mut request, lists, _) = host.asks(&classes, false);
        request.facts = RequestFacts {
            ring: Some(&ring),
            ring_positions: &positions,
            ..keyed()
        };
        let now = Instant::now();
        host.picked(&lists.decide(&host.engine, None, now, &mut request, None))
    };
    assert_eq!(placed(0), (2, 1));
    assert_eq!(placed(3), (0, 2));
}

/// Each side of a pair whose two sides read the lent ring by two words
/// is placed from its own places: the prefill side from the request's
/// list, the decode side from the list stated for it
/// ([`RequestFacts::decode_ring_positions`]), as each of the gateway's
/// two policies walks the pair's one ring from its own place
/// (`model_gateway/src/routers/common/placement.rs`, lines 3016 to 3031
/// and 3098 to 3101 at a06805492). The plain ring ahead of the prefix
/// form, and the reverse. A decode side stated no list of its own reads
/// the request's one list. One stated an empty list is placed by its
/// stage's own ring, and the prefill side keeps its place.
#[test]
fn each_side_of_a_pair_of_two_ring_words_places_from_its_own_places() {
    let address = |at: &str| format!("grpc://{at}:9000");
    let model = "decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2";
    let ring = InOrder(model.split(' ').map(address).collect());
    let (plain, prefix) = ("consistent_hashing", "prefix_hash");
    for (prefill, decode) in [(plain, prefix), (prefix, plain)] {
        let document = PolicySpec::pair_presets(prefill, decode);
        let spec = AppliedSpec::validate(&document.expect("the pair presets"));
        let host = Host::under((spec.expect("a valid document"), Arc::default()), 3, 3);
        let placed = |position: u64, decode_ring_positions: Option<&[u64]>| {
            let classes = every(3, 3);
            let positions = [position];
            // The host's own request over its two lists, every pair open.
            let Asks(mut request, lists, _) = host.asks(&classes, false);
            request.facts = RequestFacts {
                ring: Some(&ring),
                ring_positions: &positions,
                decode_ring_positions,
                ..keyed()
            };
            let now = Instant::now();
            host.picked(&lists.decide(&host.engine, None, now, &mut request, None))
        };
        // The prefill from the first place and the decode from the
        // fourth, then the reverse.
        assert_eq!(placed(0, Some(&[3])), (2, 2), "{prefill}, {decode}");
        assert_eq!(placed(3, Some(&[0])), (0, 1), "{prefill}, {decode}");
        assert_eq!(placed(0, None), (2, 1), "{prefill}, {decode}");
        assert_eq!(placed(0, Some(&[])).0, 2, "{prefill}, {decode}");
    }
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

fn completed(side: Side, served: u64) -> Said {
    Said::Completed(outcome(side, WorkerId(served)))
}

/// The pick's target at `index`.
fn target_at(decision: &Decision, index: usize) -> Target {
    let pick = decision.pick().expect("a pick");
    *pick.targets.as_slice().get(index).expect("a target")
}

/// A pick of one prefill (worker 1) with two decodes (workers 101 and
/// 102): the first target and an alternate, as the pair entry orders
/// them.
fn two_decodes() -> Decision {
    let mut targets = Targets::new();
    for (at, decode) in [(0, 101), (1, 102)] {
        targets.push(Target {
            position: 0,
            id: WorkerId(1),
            dp_rank: 0,
            pair: Some(Pair {
                decode_position: at,
                decode_id: WorkerId(decode),
                ..Pair::default()
            }),
        });
    }
    Decision::Pick(Pick {
        targets,
        ..Pick::default()
    })
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

/// One word of a held pair told to the stages, the clock reading 9,900
/// and every worker still listed.
fn tell(engine: &Engine, held: &HeldDecision<'_>, word: Said) {
    lease_events::pair(&lease_events::of_host(engine, engine), held, word, |_| {
        9_900
    });
}

/// Each word of a pair, both sides issued with the prefill, the host
/// taking the alternate decode. A side's dispatch names the worker it
/// booked at its issue and the worker it goes to, and its end names
/// the worker it went to.
#[test]
fn each_word_of_a_pair_is_told_under_its_sides_lease() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let decision = two_decodes();
    let issued = issue(engine, &keyed(), &decision, LeaseId(9), false);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    let own = lease.session;
    let taken = target_at(&decision, 1);
    let held = HeldDecision {
        decision: &decision,
        lease: &lease,
        index: 1,
        target: &taken,
        decode: HeldDecode {
            session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: false,
        },
    };
    tell(engine, &held, dispatched(Side::Prefill, 1));
    tell(engine, &held, dispatched(Side::Decode, 1));
    assert_eq!(
        pair.words(),
        [
            ("dispatched", Side::Prefill, 1, 1, own, 7_000, 1),
            ("dispatched", Side::Decode, 101, 1, session, 7_000, 102),
        ]
    );
    tell(engine, &held, completed(Side::Prefill, 1));
    tell(engine, &held, completed(Side::Decode, 102));
    assert_eq!(
        pair.words(),
        [
            ("completed", Side::Prefill, 1, 0, own, 7_000, 1),
            ("completed", Side::Decode, 102, 1, session, 7_000, 102),
        ]
    );
    // A side a pair's lease does not hold is told to nobody.
    tell(engine, &held, completed(Side::Encode, 1));
    tell(engine, &held, dispatched(Side::Single, 0));
    assert!(pair.words().is_empty());
}

/// Each side's lease states its own side's session verdict. The
/// prefill's is the pick's. The decode's is the one the pair of the
/// pick's first target keeps, whatever the prefill's is, and a claim
/// names that pair's decode.
#[test]
fn each_side_of_a_pair_states_its_own_session_verdict() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let stated = |decode_verdict| {
        let mut targets = Targets::new();
        targets.push(Target {
            id: WorkerId(1),
            pair: Some(Pair {
                decode_id: WorkerId(101),
                decode_verdict,
                ..Pair::default()
            }),
            ..Target::default()
        });
        let decision = Decision::Pick(Pick {
            targets,
            session_verdict: SessionVerdict::Pinned,
            ..Pick::default()
        });
        let issued = issue(engine, &keyed(), &decision, LeaseId(9), true);
        let (prefill, _) = issued.expect("a pair's lease");
        let decode = issue_decode(engine, &keyed(), &decision, &prefill, 9_900);
        let decode = decode.expect("the decode side's lease");
        (prefill.session_verdict, decode.session_verdict)
    };
    let claimed = SessionVerdict::Claimed(WorkerId(101));
    for (kept, whole) in [
        (DecodeVerdict::NONE, SessionVerdict::None),
        (DecodeVerdict::PINNED, SessionVerdict::Pinned),
        (DecodeVerdict::RESPILL, SessionVerdict::Respill),
        (DecodeVerdict::CLAIMED, claimed),
    ] {
        assert_eq!(stated(kept), (SessionVerdict::Pinned, whole));
    }
}

#[test]
fn a_release_ends_the_side_it_names_and_with_none_both() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let issued = issue(engine, &keyed(), &pair.decision, LeaseId(9), false);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    let own = lease.session;
    let taken = target_at(&pair.decision, 0);
    let held = HeldDecision {
        decision: &pair.decision,
        lease: &lease,
        index: 0,
        target: &taken,
        decode: HeldDecode {
            session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: false,
        },
    };
    let prefill = ("released", Side::Prefill, 1, 0, own, 7_000, 0);
    let decode = ("released", Side::Decode, 101, 0, session, 7_000, 0);
    let gone = ReleaseReason::ClientGone;
    tell(
        engine,
        &held,
        Said::Released(None, ReleaseReason::CommitFailed),
    );
    assert_eq!(pair.words(), [prefill, decode]);
    tell(engine, &held, Said::Released(Some(Side::Decode), gone));
    assert_eq!(pair.words(), [decode]);
    tell(engine, &held, Said::Released(Some(Side::Prefill), gone));
    assert_eq!(pair.words(), [prefill]);
}

/// A decode that stood chosen goes out as its pick's with no call. Its
/// dispatch is its issue too: the stages hear the issue first, dated at
/// the dispatch, and both words name the decode it goes to.
#[test]
fn a_chosen_decode_sent_with_no_call_is_told_issued_at_its_dispatch() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let decision = two_decodes();
    let issued = issue(engine, &keyed(), &decision, LeaseId(9), true);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    let taken = target_at(&decision, 1);
    let held = HeldDecision {
        decision: &decision,
        lease: &lease,
        index: 1,
        target: &taken,
        decode: HeldDecode {
            session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    tell(engine, &held, dispatched(Side::Decode, 1));
    assert_eq!(
        pair.words(),
        [
            ("issued", Side::Decode, 102, 1, session, 9_900, 1),
            ("dispatched", Side::Decode, 102, 1, session, 7_000, 102),
        ]
    );
}

/// A decode that stands chosen was issued to no stage, so its end with
/// its prefill tells the prefill's alone.
#[test]
fn a_decode_that_stands_chosen_ends_with_its_prefill_and_is_told_to_nobody() {
    let pair = decided(1);
    let engine = pair.spec.engine();
    let issued = issue(engine, &keyed(), &pair.decision, LeaseId(9), true);
    let (lease, session) = issued.expect("a pair's lease");
    pair.words();
    let own = lease.session;
    let taken = target_at(&pair.decision, 0);
    let held = HeldDecision {
        decision: &pair.decision,
        lease: &lease,
        index: 0,
        target: &taken,
        decode: HeldDecode {
            session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    tell(
        engine,
        &held,
        Said::Released(None, ReleaseReason::Abandoned),
    );
    tell(
        engine,
        &held,
        Said::Released(Some(Side::Decode), ReleaseReason::Abandoned),
    );
    tell(engine, &held, completed(Side::Decode, 101));
    assert_eq!(
        pair.words(),
        [("released", Side::Prefill, 1, 0, own, 7_000, 0)]
    );
}

/// The engine decides a pair whole for the guard that asks. A pick is
/// issued with both sides told, and each side's dispatch and end reach
/// the stages from what the guard lends. A request held to a wire no
/// worker of the lists is dialed on is declined under the wires' name.
#[test]
fn the_engine_decides_a_pair_whole_and_hears_each_sides_words_from_its_guard() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let (asks, now) = (host.asks(&classes, false), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    let first = target_at(guard.decision(), 0);
    let decode = first.pair.expect("a pair").decode_id;
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, first.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("issued", Side::Decode, decode, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
    let mut undialed = host.asks(&classes, false);
    undialed.0.constraints.wire_modes = WireModes::ZMQ;
    let decided = host.engine.decide_under(GatewayId::LOCAL, &undialed, now);
    match decided
        .expect("an answer")
        .whole()
        .expect("a whole decision")
        .0
        .decision
    {
        Decision::Shed(shed) => assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::PolicyDeclined, WIRES)
        ),
        other => panic!("not shed: {other:?}"),
    }
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 0));
}

/// A pair's decision alone ([`decide`]) is the decision its leasing form
/// makes ([`decided`]): of two engines made alike, the one asked for the
/// decision alone names, call by call, the pair the other issues a lease
/// for, and behind a held target the decode the other issues. It takes
/// no lease id and tells no stage. A request held to a wire no worker
/// of the lists is dialed on is declined under the wires' name by the
/// decision alone, and the leasing form answers that decline with no
/// lease, takes no id for it and tells no stage.
#[test]
fn a_pairs_decision_alone_is_its_leasing_forms_and_takes_no_id_and_tells_no_stage() {
    let (alone, leasing) = (Host::of(2, 2), Host::of(2, 2));
    let classes = every(2, 2);
    let now = Instant::now();
    for _ in 0..4 {
        let mut asked = alone.asks(&classes, false);
        let mut leased = leasing.asks(&classes, false);
        let decision = decide(
            alone.engine.loaded(),
            &asked.1,
            &mut asked.0,
            None,
            None,
            now,
            |_| true,
            SessionCountedUnder::default,
        );
        let outcome = super::decided(
            leasing.engine.loaded(),
            &leased.1,
            &mut leased.0,
            None,
            None,
            now,
            |_| true,
            SessionCountedUnder::default,
            &|_| None,
        );
        assert!(outcome.lease.is_some(), "the leasing form issues its pick");
        assert_eq!(alone.picked(&decision), leasing.picked(&outcome.decision));
    }
    // A pair whose decode is sent later, and that decode decided again
    // behind the target each engine's first pick names.
    let mut asked = alone.asks(&classes, true);
    let mut leased = leasing.asks(&classes, true);
    let first = decide(
        alone.engine.loaded(),
        &asked.1,
        &mut asked.0,
        None,
        None,
        now,
        |_| true,
        SessionCountedUnder::default,
    );
    let outcome = super::decided(
        leasing.engine.loaded(),
        &leased.1,
        &mut leased.0,
        None,
        None,
        now,
        |_| true,
        SessionCountedUnder::default,
        &|_| None,
    );
    assert_eq!(alone.picked(&first), leasing.picked(&outcome.decision));
    let (ours, theirs) = (target_at(&first, 0), target_at(&outcome.decision, 0));
    let lease = outcome.lease.expect("the pair's lease").lease;
    let held = HeldDecision {
        decision: &outcome.decision,
        lease: &lease,
        index: 0,
        target: &theirs,
        decode: HeldDecode {
            session: outcome.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    let call = now + Duration::from_millis(1);
    let again = decide(
        alone.engine.loaded(),
        &asked.1,
        &mut asked.0,
        Some(&ours),
        None,
        call,
        |_| true,
        SessionCountedUnder::default,
    );
    let issued = super::decided(
        leasing.engine.loaded(),
        &leased.1,
        &mut leased.0,
        Some(&held),
        None,
        call,
        |_| true,
        SessionCountedUnder::default,
        &|_| None,
    );
    assert!(issued.lease.is_some(), "the leasing form issues the decode");
    assert_eq!(alone.picked(&again), leasing.picked(&issued.decision));
    assert_eq!(alone.picked(&again).0, alone.picked(&first).0);
    // A request held to a wire no worker of the lists is dialed on.
    let mut undialed = alone.asks(&classes, false);
    undialed.0.constraints.wire_modes = WireModes::ZMQ;
    match decide(
        alone.engine.loaded(),
        &undialed.1,
        &mut undialed.0,
        None,
        None,
        now,
        |_| true,
        SessionCountedUnder::default,
    ) {
        Decision::Shed(shed) => assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::PolicyDeclined, WIRES)
        ),
        other => panic!("not shed: {other:?}"),
    }
    // The leasing form answers that request the same decline, first and
    // behind the held decision, with no lease and no session.
    let mut refused = leasing.asks(&classes, true);
    refused.0.constraints.wire_modes = WireModes::ZMQ;
    for behind in [None, Some(&held)] {
        let declined = super::decided(
            leasing.engine.loaded(),
            &refused.1,
            &mut refused.0,
            behind,
            None,
            call,
            |_| true,
            SessionCountedUnder::default,
            &|_| None,
        );
        assert!(matches!(&declined.decision, Decision::Shed(shed) if shed.emptied_by == WIRES));
        assert!(declined.lease.is_none());
        assert_eq!(declined.decode_session, Session::NONE);
    }
    // The leasing form took an id a pair and told each side's issue, and
    // neither for a decline: the decision alone took none and told nobody.
    assert_eq!(leasing.engine.leases.0.load(Ordering::Relaxed), 5);
    assert_eq!(leasing.words().len(), 10);
    assert_eq!(alone.engine.leases.0.load(Ordering::Relaxed), 0);
    assert!(alone.words().is_empty(), "no stage heard a word");
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

/// The entry that issues no lease decides a pair as the leasing entry
/// does ([`InProcess::decide_without_lease`]). Of two engines made
/// alike it names, call by call, the targets the other issues a lease
/// for, and it takes no lease id and tells no stage. The decode asked
/// again behind the target a first decision names
/// ([`InProcess::decide_decode_without_lease`]) is the decode the
/// leasing entry answers a guard, over the lists as they stand at the
/// call: the decode the first decision named is not ready by then.
#[test]
fn a_pair_without_a_lease_is_decided_as_the_leasing_entry_decides_it() {
    let (bare, leasing) = (Host::of(2, 3), Host::of(2, 3));
    let classes = every(2, 3);
    let now = Instant::now();
    for call in 0..6 {
        let asks = bare.asks(&classes, false);
        let decision = bare.engine.decide_without_lease(&asks, now);
        let (decision, _) = decision.expect("a decision");
        let outcome = leased(&leasing, &leasing.asks(&classes, false), now);
        assert!(outcome.lease.is_some(), "the leasing entry issues its pick");
        assert!(!named(&decision).is_empty(), "call {call}: a pick");
        assert_eq!(named(&decision), named(&outcome.decision), "call {call}");
    }
    // A pair whose decode is sent later, then that decode decided again
    // behind the target each engine's first decision names.
    let asks = bare.asks(&classes, true);
    let first = bare.engine.decide_without_lease(&asks, now);
    let (first, _) = first.expect("a decision");
    let theirs = leasing.asks(&classes, true);
    let outcome = leased(&leasing, &theirs, now);
    assert_eq!(named(&first), named(&outcome.decision));
    let (prefill, chosen) = bare.picked(&first);
    for host in [&bare, &leasing] {
        host.decodes[chosen].set_status(WorkerStatus::NotReady);
    }
    let (ours, target) = (target_at(&first, 0), target_at(&outcome.decision, 0));
    let lease = outcome.lease.expect("the pair's lease").lease;
    let held = HeldDecision {
        decision: &outcome.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: outcome.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    let call = now + Duration::from_millis(1);
    let again = bare.engine.decide_decode_without_lease(&asks, &ours, call);
    let again = again.expect("a decision");
    let issued = leasing
        .engine
        .decide_decode_under(GatewayId::LOCAL, &theirs, &held, call);
    let issued = issued.expect("a decision");
    assert!(
        issued.lease.is_some(),
        "the leasing entry issues the decode"
    );
    assert_eq!(named(&again), named(&issued.decision));
    let (kept, decode) = bare.picked(&again);
    assert_eq!(kept, prefill, "the held prefill");
    assert_ne!(decode, chosen, "a decode that is ready at the call");
    // The leasing entry took an id a pair and told each side's issue.
    // The other took none and told nobody.
    assert_eq!(leasing.engine.leases.0.load(Ordering::Relaxed), 7);
    assert_eq!(leasing.words().len(), 14);
    assert_eq!(bare.engine.leases.0.load(Ordering::Relaxed), 0);
    assert!(bare.words().is_empty(), "no stage heard a word");
}

/// The entry that issues no lease keeps the test of a request's wires
/// that a pair's decision makes ([`decide`]), at the pair's first
/// decision and at its decode's behind a held target. The prefills
/// here are dialed on one wire and the decodes on another. A request
/// that states one of the two is declined under the wires' name: a
/// worker of the other list is dialed on a wire it does not state. A
/// request that states both is decided, and so is one that states a
/// third beside them and one that states none. The leasing entry
/// answers each request alike.
#[test]
fn a_pair_without_a_lease_keeps_the_test_of_its_wires() {
    let dialed = |name: &str, worker_type, mode| -> Arc<[Arc<dyn Worker>]> {
        let spec = |at: usize| WorkerSpec {
            worker_type,
            connection_mode: mode,
            ..WorkerSpec::new(format!("http://{name}-{at}:9000"))
        };
        let ready = |at| {
            let worker = <BasicWorkerBuilder>::from_spec(spec(at))
                .health_config(no_health_check())
                .build();
            worker.set_status(WorkerStatus::Ready);
            Arc::new(worker) as Arc<dyn Worker>
        };
        (0..2).map(ready).collect()
    };
    let host = Host::over(
        hearing(),
        dialed("prefill", WorkerType::Prefill, ConnectionMode::Http),
        dialed("decode", WorkerType::Decode, ConnectionMode::Grpc),
    );
    let [prefills, decodes] = &host.lists;
    let wires = |list: &PublishedList| host.engine.members_of(list.lent()).wires();
    assert_eq!(
        (wires(prefills), wires(decodes)),
        (WireModes::HTTP, WireModes::GRPC)
    );
    let classes = every(2, 2);
    let now = Instant::now();
    let stating = |wires: WireModes| {
        let mut asks = host.asks(&classes, true);
        asks.0.constraints.wire_modes = wires;
        asks
    };
    let both = WireModes::HTTP.union(WireModes::GRPC);
    let first = host.engine.decide_without_lease(&stating(both), now);
    let held = target_at(&first.expect("a decision").0, 0);
    for wires in [WireModes::NONE, both, both.union(WireModes::ZMQ)] {
        let asks = stating(wires);
        let decision = host.engine.decide_without_lease(&asks, now);
        let (decision, _) = decision.expect("a decision");
        assert!(decision.pick().is_some(), "{wires:?}: {decision:?}");
        let again = host.engine.decide_decode_without_lease(&asks, &held, now);
        let again = again.expect("a decision");
        assert!(again.pick().is_some(), "{wires:?} again: {again:?}");
        assert!(leased(&host, &asks, now).decision.pick().is_some());
    }
    let declined = (ShedReason::PolicyDeclined, WIRES);
    for wires in [WireModes::HTTP, WireModes::GRPC, WireModes::ZMQ] {
        let asks = stating(wires);
        let decision = host.engine.decide_without_lease(&asks, now);
        assert_eq!(shed_for(&decision.expect("a decision").0), declined);
        let again = host.engine.decide_decode_without_lease(&asks, &held, now);
        assert_eq!(shed_for(&again.expect("a decision")), declined);
        assert_eq!(shed_for(&leased(&host, &asks, now).decision), declined);
    }
}

/// A request that hands no pair's lists is decided over no worker by
/// the entries that issue no lease, as the leasing entries answer it: a
/// pair's prefill, and its decode asked again behind a held target.
#[test]
fn a_pair_without_a_lease_that_hands_no_lists_is_decided_over_no_worker() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let now = Instant::now();
    let asks = host.asks(&classes, true);
    let outcome = leased(&host, &asks, now);
    let target = target_at(&outcome.decision, 0);
    let lease = outcome.lease.expect("the pair's lease").lease;
    let held = HeldDecision {
        decision: &outcome.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: outcome.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    // The request by itself is a maker that hands no list.
    let alone: &Request<'_> = &asks.0;
    let none = (ShedReason::NoEndpoints, "none");
    let prefill = host.engine.decide_without_lease(alone, now);
    assert_eq!(shed_for(&prefill.expect("a decision").0), none);
    assert_eq!(shed_for(&leased(&host, alone, now).decision), none);
    let again = host.engine.decide_decode_without_lease(alone, &target, now);
    assert_eq!(shed_for(&again.expect("a decision")), none);
    let issued = host
        .engine
        .decide_decode_under(GatewayId::LOCAL, alone, &held, now);
    let issued = issued.expect("an answer");
    assert_eq!(shed_for(&issued.decision), none);
    assert!(issued.lease.is_none(), "a shed issues nothing");
}

/// A guard lends its pair's session of the decode phase with the decode
/// side ([`HeldDecode::session`]), and every word of that side is told
/// under it: the session the side's issue stated, which is not the
/// prefill side's.
#[test]
fn a_guards_decode_side_is_told_under_the_session_of_its_issue() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let (asks, now) = (host.asks(&classes, false), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    let decode = target_at(guard.decision(), 0).pair.expect("a pair");
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Decode, decode.decode_id), now);
    drop(guard);
    let heard = std::mem::take(&mut *host.heard.lock());
    let told = |side: Side| -> Vec<(&'static str, Session)> {
        let of_side = heard.iter().filter(|word| word.1 == side);
        of_side.map(|word| (word.0, word.4)).collect()
    };
    let own = asks.0.facts.session();
    let decode_side = told(Side::Decode);
    let issued = decode_side.first().expect("the decode's issue").1;
    assert_ne!(issued, own, "each side keeps its own session");
    assert_eq!(
        decode_side,
        [
            ("issued", issued),
            ("dispatched", issued),
            ("completed", issued),
        ]
    );
    assert!(told(Side::Prefill).iter().all(|word| word.1 == own));
}

/// A decode sent later stands chosen in its guard. Behind the prefill's
/// dispatch the engine decides it again over the host's lists of that
/// call. The side is issued then under the pair's lease, dated at the
/// call, and its later words bear the first decision's instant.
#[test]
fn the_engine_decides_a_chosen_decode_again_for_its_guard() {
    let host = Host::of(1, 2);
    let classes = every(1, 2);
    let (asks, now) = (host.asks(&classes, true), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    assert!(guard.decode_chosen());
    let first = target_at(guard.decision(), 0);
    let chosen = host.picked(guard.decision()).1;
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    // The decode the first pick named stops being ready behind it.
    host.decodes[chosen].set_status(WorkerStatus::NotReady);
    let call = now + Duration::from_millis(1);
    let again = *guard.decide_decode_under(&asks, call).expect("a decision");
    assert_eq!(host.picked(&again), (0, 1 - chosen));
    assert!(!guard.decode_chosen(), "the call's pick issued the side");
    let decode = target_at(&again, 0).pair.expect("a pair").decode_id;
    guard.dispatched_decode(call);
    guard.completed(&outcome(Side::Prefill, first.id), call);
    guard.completed(&outcome(Side::Decode, decode), call);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("issued", Side::Decode, decode, at + 1_000),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
}

/// A picker that picks as [`Best`] and panics at the issue of the side
/// it names.
#[derive(Debug)]
struct StopsAt(Side);

impl Lifecycle for StopsAt {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        assert!(lease.side != self.0, "this stage panics there");
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

/// [`hearing`] with a prefill side that picks through [`StopsAt`]: at
/// the decode side's issue, or at the prefill side's where `prefill`.
/// The decode profile's stage, which keeps what it hears, is told each
/// word ahead of it.
fn stopping(prefill: bool) -> (AppliedSpec, Arc<Mutex<Vec<Word>>>) {
    let stage: OwnPicker = if prefill {
        ("stops-at", || Box::new(StopsAt(Side::Prefill)))
    } else {
        ("stops-at", || Box::new(StopsAt(Side::Decode)))
    };
    hearing_with(Some(stage))
}

/// A stage that panics at the issue of a decode decided again is held
/// and counted with the words a stage did not hear. The call answers
/// its pick and its guard holds the side as issued, so the stage that
/// heard the issue ahead of the panic hears the side's dispatch and its
/// end. Before, the call answered a fault, its guard kept the first
/// decode as chosen, and no word ended the issue that stage had heard.
///
/// The gateway's own routing picks a pair's decode with its prefill and
/// has no stage that hears an issue
/// (`model_gateway/src/routers/common/placement.rs:1227`).
#[test]
fn a_stage_that_panics_at_a_decode_calls_issue_costs_the_call_nothing() {
    let host = Host::under(stopping(false), 1, 2);
    let classes = every(1, 2);
    let (asks, now) = (host.asks(&classes, true), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    assert!(guard.decode_chosen());
    let first = target_at(guard.decision(), 0);
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    let call = now + Duration::from_millis(1);
    let again = *guard.decide_decode_under(&asks, call).expect("a decision");
    let decode = target_at(&again, 0).pair.expect("a pair").decode_id;
    assert!(!guard.decode_chosen(), "the call's pick issued the side");
    guard.dispatched_decode(call);
    guard.completed(&outcome(Side::Prefill, first.id), call);
    guard.completed(&outcome(Side::Decode, decode), call);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("issued", Side::Decode, decode, at + 1_000),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
    // The held panic alone is counted, and no later word of the side.
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
}

/// A stage that panics at an issue of a pair's first decision is held
/// and counted with the words a stage did not hear. The decision
/// answers its pick and its lease, so the guard holds both sides as
/// issued: the stage that heard each issue ahead of the panic hears
/// each side's dispatch and its end. Before, the request was answered
/// a fault and no word ended the two issues that stage had heard.
///
/// The classic pass tells a policy nothing where a pick is issued: it
/// asks each side's policy for one pick
/// (`model_gateway/src/routers/common/placement.rs:2758`, `:2820`).
#[test]
fn a_stage_that_panics_at_a_pairs_first_issue_costs_the_request_nothing() {
    let host = Host::under(stopping(false), 1, 2);
    let classes = every(1, 2);
    // The decode is issued with its prefill: its issue is the one the
    // stage panics at, behind the prefill side's.
    let (asks, now) = (host.asks(&classes, false), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    assert!(!guard.decode_chosen(), "the pick issued both sides");
    let first = target_at(guard.decision(), 0);
    let decode = first.pair.expect("a pair").decode_id;
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, first.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("issued", Side::Decode, decode, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
    // The held panic alone is counted: no decision faulted.
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
}

/// Where the stage panics at the prefill side's issue, the one catch
/// holds both issues: the stage ahead of it heard the prefill's, and
/// no stage hears the decode side's. The decision answers its pick and
/// its lease all the same, so the guard holds both sides as issued,
/// and a stage hears the decode side's dispatch and its end with no
/// issue ahead of them, as the lifecycle's doc says a stage may.
#[test]
fn a_stage_that_panics_at_a_pairs_prefill_issue_leaves_the_decode_side_untold() {
    let host = Host::under(stopping(true), 1, 2);
    let classes = every(1, 2);
    let (asks, now) = (host.asks(&classes, false), Instant::now());
    let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now)
        .expect("a decision");
    assert!(!guard.decode_chosen(), "the pick issued both sides");
    let first = target_at(guard.decision(), 0);
    let decode = first.pair.expect("a pair").decode_id;
    let at = guard.lease().expect("a lease").lease.issued_micros;
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, first.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    drop(guard);
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, first.id, at),
            ("dispatched", Side::Prefill, first.id, at),
            ("dispatched", Side::Decode, decode, at),
            ("completed", Side::Prefill, first.id, at),
            ("completed", Side::Decode, decode, at),
        ]
    );
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
}

/// The prompt size each word of a pair states to a stage, through a
/// guard of `door`: the prefill side's words, then the decode side's.
/// The first decision is `first`'s. Behind the prefill's dispatch the
/// decode is decided again by `again`, or with none goes out as the
/// first decision left it. The decode side is then sent and ends, or is
/// `released` with no dispatch. The guard is dropped with its prefill
/// sent, so that side ends as cancelled.
fn sizes_by_side<D: DecisionEngine, M: RequestMaker>(
    door: &D,
    gateway: GatewayId,
    first: &M,
    again: Option<&M>,
    released: bool,
) -> [Vec<(&'static str, u32)>; 2] {
    // Words an earlier guard said on this thread are not this one's.
    SIZES.with_borrow_mut(Vec::clear);
    let now = Instant::now();
    let mut guard = DecisionGuard::decide_under(door, gateway, first, now).expect("a decision");
    guard.dispatched(now);
    if let Some(again) = again {
        let picked = guard.decide_decode_under(again, now);
        assert!(picked.is_ok_and(|decision| decision.pick().is_some()));
    }
    if released {
        guard.released_decode(ReleaseReason::ClientGone, now);
    } else {
        let decode = target_at(guard.decision(), 0).pair.expect("a pair");
        guard.dispatched_decode(now);
        guard.completed(&outcome(Side::Decode, decode.decode_id), now);
    }
    drop(guard);
    let heard = sizes();
    let of = |side: Side| {
        let words = heard.iter().filter(|word| word.1 == side);
        words.map(|&(word, _, size)| (word, size)).collect()
    };
    [of(Side::Prefill), of(Side::Decode)]
}

/// A decode call may state another prompt size than the first decision
/// did. The side is issued under the call's size, and its dispatch, its
/// end and its release state that size too: a stage that books a size
/// at the issue reads the same size at every later word of the side.
/// The prefill side states the first decision's size behind the call
/// as before it. A decode that goes out with no call states the first
/// decision's size, issued at its own dispatch or with its prefill.
#[test]
fn a_decode_calls_side_states_the_calls_size_at_every_word() {
    let host = Host::of(1, 2);
    let classes = every(1, 2);
    let first = host.asks(&classes, true);
    let stated = first.0.facts.prompt_tokens;
    let mut again = host.asks(&classes, true);
    again.0.facts.prompt_tokens = stated + 7;
    let (door, local, called) = (&host.engine, GatewayId::LOCAL, stated + 7);
    let own = [
        ("issued", stated),
        ("dispatched", stated),
        ("completed", stated),
    ];
    let [prefill, decode] = sizes_by_side(door, local, &first, Some(&again), false);
    assert_eq!(prefill, own);
    assert_eq!(
        decode,
        [
            ("issued", called),
            ("dispatched", called),
            ("completed", called),
        ]
    );
    let [_, decode] = sizes_by_side(door, local, &first, Some(&again), true);
    assert_eq!(decode, [("issued", called), ("released", called)]);
    let [_, decode] = sizes_by_side(door, local, &first, None, false);
    assert_eq!(decode, own);
    let with = host.asks(&classes, false);
    let [_, decode] = sizes_by_side(door, local, &with, None, false);
    assert_eq!(decode, own);
}

/// A decode call may state another text than the first decision did.
/// The side is issued under the call's three sizes, and its dispatch,
/// its end and its release state them too: a stage that books a
/// request of text by its characters reads one length at every word of
/// the side, and a plugin is shown the size the call's own request
/// showed. Neither request here states a token count: the first is a
/// text of 400 characters in 800 bytes, the call's one of 900 in 1,800.
/// The prefill side states the first decision's lengths behind the
/// call as before it, and so does a decode that goes out with no call.
#[test]
fn a_decode_calls_side_states_the_calls_text_lengths_at_every_word() {
    let host = Host::of(1, 2);
    let classes = every(1, 2);
    let of_text = |chars, bytes| {
        let mut asks = host.asks(&classes, true);
        let facts = &mut asks.0.facts;
        (facts.prompt_tokens, facts.prompt_chars, facts.prompt_bytes) = (0, chars, bytes);
        asks
    };
    let (first, again) = (of_text(400, 800), of_text(900, 1_800));
    let (door, local) = (&host.engine, GatewayId::LOCAL);
    // What each word of a side stated: the prefill's, then the decode's.
    let heard = |again, released| {
        LENGTHS.with_borrow_mut(Vec::clear);
        let _ = sizes_by_side(door, local, &first, again, released);
        let heard = LENGTHS.with_borrow_mut(std::mem::take);
        let of = |side| {
            let words = heard.iter().filter(|word| word.0 == side);
            words.map(|word| word.1).collect::<Vec<_>>()
        };
        [of(Side::Prefill), of(Side::Decode)]
    };
    let (own, called) = ([400, 800, 200], [900, 1_800, 450]);
    let [prefill, decode] = heard(Some(&again), false);
    assert_eq!(prefill, [own; 3], "issued, sent and ended");
    assert_eq!(decode, [called; 3], "issued, sent and ended");
    let [_, decode] = heard(Some(&again), true);
    assert_eq!(decode, [called; 2], "issued and released");
    let [_, decode] = heard(None, false);
    assert_eq!(decode, [own; 3], "with no call");
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

/// The engine's own task, turned once: it takes what stands ready for
/// it, here the newest snapshot on the channel its host handed.
async fn turned(engine: &InProcess) {
    let step = Duration::from_millis(1);
    assert!(tokio::time::timeout(step, engine.run()).await.is_err());
}

/// Each side of a pair that scores by load reads its workers' reports in
/// the snapshot its host hands for the call: the prefill and the decode
/// whose queue is empty are picked at every seed. The decode decided
/// again behind its prefill reads the snapshot of its own call, and
/// keeps the prefill that was sent whatever that snapshot says of the
/// prefills. Handed no snapshot, the same engine reads no report: no
/// worker here holds a request in flight, so each scores alike and the
/// picks spread over both lists. The gateway's classic router reads the
/// reports where a side's policy has workers to choose between: each
/// side of a pair asks its policy
/// (`model_gateway/src/routers/common/placement.rs:1521` and `:1583`),
/// and `least_load` reads its snapshot and each worker's report in it
/// (`model_gateway/src/policies/least_load.rs:918` and `:450`, at main
/// `ddd8a44cb`), behind its answer for a side's one healthy worker
/// (`:1119-1120`). A decode pool all over its overload thresholds is not
/// asked of its policy there: its pick is the least loaded partner
/// (`placement.rs:1575-1578`). No worker of this case is over one.
#[test]
fn a_pair_that_scores_by_load_reads_the_reports_its_host_hands() {
    let (host, classes) = of_presets("least_load", "least_load");
    let mut unread = [[0_u32; 4]; 2];
    for seed in 0..64 {
        let fed = host.asks_by(&classes, (seed, false), Some(queues(2, 1)));
        assert_eq!(host.first(&fed), (2, 1), "seed {seed}");
        let (prefill, decode) = host.first(&host.asks_by(&classes, (seed, false), None));
        unread[0][prefill] += 1;
        unread[1][decode] += 1;
    }
    assert!(
        unread.iter().flatten().all(|&picks| picks > 0),
        "{unread:?}"
    );
    // The decode decided again: in its call's snapshot another decode
    // has the empty queue, and another prefill too. The prefill that
    // was sent stays the pair's.
    for seed in 0..16 {
        let now = Instant::now();
        let sent = host.asks_by(&classes, (seed, true), Some(queues(2, 1)));
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &sent, now)
            .expect("a decision");
        assert_eq!(host.picked(guard.decision()), (2, 1), "seed {seed}");
        guard.dispatched(now);
        let call = host.asks_by(&classes, (seed, true), Some(queues(0, 3)));
        let again = *guard.decide_decode_under(&call, now).expect("a decision");
        assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
    }
}

/// The snapshot reaches a side that scores by load beside a side that
/// reads no report. With round robin prefills the decode whose queue is
/// empty is picked at every seed, and with round robin decodes the
/// prefill whose queue is empty.
#[test]
fn a_side_that_scores_by_load_reads_the_reports_beside_a_side_that_reads_none() {
    for (prefill, decode) in [("round_robin", "least_load"), ("least_load", "round_robin")] {
        let (host, classes) = of_presets(prefill, decode);
        for seed in 0..64 {
            let fed = host.asks_by(&classes, (seed, false), Some(queues(2, 1)));
            let picked = host.first(&fed);
            if decode == "least_load" {
                assert_eq!(picked.1, 1, "the decode at seed {seed}");
            } else {
                assert_eq!(picked.0, 2, "the prefill at seed {seed}");
            }
        }
    }
}

/// A pair whose request gives no snapshot reads the newest the engine's
/// own task keeps at hand, as a single request does. At its first
/// decision and at its decode decided again, the worker whose queue is
/// empty in the reports the load monitor published last is picked at
/// every seed, by the entries that issue a lease and by those that
/// issue none. Until the task has read the channel the engine holds no
/// snapshot, and the picks spread. A snapshot the request gives is read
/// in place of the engine's.
#[tokio::test(start_paused = true)]
async fn a_pair_whose_request_gives_no_snapshot_reads_the_engines_newest() {
    let (host, classes) = of_presets("least_load", "least_load");
    let mut channel = None;
    let feed = TestFeed::attached_to(|loads| channel = loads);
    // The load monitor's side: the reports published again.
    let publish = |prefill, decode| {
        let published = reports(prefill, decode);
        feed.publish(&published.into_iter().collect());
    };
    publish(2, 1);
    host.engine.set_loads(channel.expect("the feed's channel"));
    let bare = |seed, later| host.asks_by(&classes, (seed, later), None);
    let mut unread = [[0_u32; 4]; 2];
    for seed in 0..64 {
        let (prefill, decode) = host.first(&bare(seed, false));
        unread[0][prefill] += 1;
        unread[1][decode] += 1;
    }
    assert!(
        unread.iter().flatten().all(|&picks| picks > 0),
        "{unread:?}"
    );
    turned(&host.engine).await;
    for seed in 0..32 {
        assert_eq!(host.first(&bare(seed, false)), (2, 1), "seed {seed}");
        let given = host.asks_by(&classes, (seed, false), Some(queues(0, 3)));
        assert_eq!(host.first(&given), (0, 3), "seed {seed}");
    }
    // The decode decided again reads what the monitor published since
    // the first decision, and keeps the prefill that was sent.
    for seed in 0..16 {
        publish(2, 1);
        turned(&host.engine).await;
        let now = Instant::now();
        let sent = bare(seed, true);
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &sent, now)
            .expect("a decision");
        assert_eq!(host.picked(guard.decision()), (2, 1), "seed {seed}");
        guard.dispatched(now);
        let unleased = host.engine.decide_without_lease(&sent, now);
        let (unleased, _) = unleased.expect("a decision");
        assert_eq!(host.picked(&unleased), (2, 1), "seed {seed}");
        publish(0, 3);
        turned(&host.engine).await;
        let call = bare(seed, true);
        let again = *guard.decide_decode_under(&call, now).expect("a decision");
        assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
        let held = target_at(&unleased, 0);
        let again = host.engine.decide_decode_without_lease(&call, &held, now);
        assert_eq!(
            host.picked(&again.expect("a decision")),
            (2, 3),
            "seed {seed}"
        );
    }
}

/// The entries that issue no lease read the reports as the leasing
/// entries do ([`InProcess::decide_without_lease`],
/// [`InProcess::decide_decode_without_lease`]). At every seed a
/// pair's first decision picks the two workers whose queues are empty
/// in the snapshot its request gives. Its decode decided again picks
/// the decode whose queue is empty in the snapshot of its own call,
/// behind the prefill the first decision named.
#[test]
fn a_pair_decided_without_a_lease_reads_the_reports_its_host_hands() {
    let (host, classes) = of_presets("least_load", "least_load");
    let now = Instant::now();
    for seed in 0..32 {
        let asks = host.asks_by(&classes, (seed, true), Some(queues(2, 1)));
        let first = host.engine.decide_without_lease(&asks, now);
        let (first, _) = first.expect("a decision");
        assert_eq!(host.picked(&first), (2, 1), "seed {seed}");
        let held = target_at(&first, 0);
        let call = host.asks_by(&classes, (seed, true), Some(queues(0, 3)));
        let again = host.engine.decide_decode_without_lease(&call, &held, now);
        let again = again.expect("a decision");
        assert_eq!(host.picked(&again), (2, 3), "seed {seed}");
    }
}

/// A pair that exists among the places is found. Six prefills, each
/// with its own decode, and the decode list held to the sixth decode's
/// place: the sixth prefill is picked at every turn. A pick among all
/// six names four of them a turn and would miss it on some. The
/// gateway's classic router does the same by the request's own rule: a
/// prefill is a candidate only where one of its partners is open for
/// the request (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1389-1394`, `:1423-1437`, at main `c738c8d3b`).
#[test]
fn a_prefill_whose_partners_stand_outside_the_decode_places_is_no_candidate() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(6, 6);
    let own: Vec<[u32; 1]> = (0..6).map(|at| [at]).collect();
    let classes = PartnerClasses::number(6, own.iter().map(|one| one.as_slice()));
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let free = Constraints::default();
    for seed in 0..24 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_records(
            spec.engine(),
            &facts,
            (&free, [unvetoed(None), unvetoed(Some(&[5]))]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        );
        assert_eq!(picked(&decision), (6, 106), "seed {seed}");
    }
}

/// What a pair's shed says is said of the prefills the request may take.
/// The one prefill place holds a worker that is not ready, the prefills
/// outside the places are ready, and every decode is over its
/// thresholds: the shed names the prefill side's stage, not the decodes'
/// veto, which stands against prefills the request may not take. The
/// gateway's classic router names the prefill role there: no prefill
/// its request's rule admits is available
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1342-1346`, `:1395-1413`, at main `c738c8d3b`).
#[test]
fn a_pair_with_no_ready_prefill_among_its_places_is_shed_at_the_prefill_stage() {
    let (spec, _) = hearing();
    let (mut prefills, mut decodes) = lists(3, 2);
    let classes = every(3, 2);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    prefills[0].health.healthy = false;
    for decode in &mut decodes {
        decode.health.overloaded = true;
    }
    let free = Constraints::default();
    let decision = decide_records(
        spec.engine(),
        &facts(),
        (&free, [unvetoed(Some(&[0])), unvetoed(None)]),
        &pool(&prefills),
        pools(&prefills, &decodes, &classes),
        &mut Scratch::default(),
    );
    match decision {
        Decision::Shed(shed) => assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::NoEndpoints, "healthy")
        ),
        other => panic!("not shed: {other:?}"),
    }
}

/// The decode asked again looks its held prefill up before it judges
/// what the call states. A prefill that is in no list reads
/// `prefill-gone` under a demand, a subset hint, places of the request's
/// own, and decode places of the list. With the prefill listed, the
/// first three read their own stage and the fourth is decided among its
/// places. The gateway's classic router asks no decode again: it selects
/// a request's pair whole, once
/// (`model_gateway/src/routers/common/placement.rs:1227-1275`, at main
/// `c738c8d3b`).
#[test]
fn a_decode_asked_for_a_prefill_that_left_reads_prefill_gone_whatever_it_states() {
    let pair = decided(2);
    let (mut prefills, mut decodes) = lists(1, 2);
    let classes = every(1, 2);
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    let again = |held: u64, cx: &Constraints<'_>, places: Option<&[u32]>| {
        decide_decode_records(
            pair.spec.engine(),
            &facts(),
            (cx, unvetoed(places)),
            &pool(&decodes),
            pools(&prefills, &decodes, &classes),
            &prefill_alone(WorkerId(held)),
            &mut Scratch::default(),
        )
    };
    let shed_of = |decision: Decision| match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("not shed: {other:?}"),
    };
    let free = Constraints::default();
    let demanded = Constraints {
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..free
    };
    let hinted = Constraints {
        subset_hint: Some(2),
        ..free
    };
    let held_to = Constraints {
        places: Some(&[0]),
        ..free
    };
    let gone = (ShedReason::NoEndpoints, "prefill-gone");
    for (cx, own) in [
        (&demanded, (ShedReason::NoEndpoints, WIRES)),
        (&hinted, (ShedReason::PolicyDeclined, SUBSET_HINT)),
        (&held_to, (ShedReason::PolicyDeclined, single::PLACES)),
    ] {
        assert_eq!(shed_of(again(5, cx, None)), gone);
        assert_eq!(shed_of(again(1, cx, None)), own);
    }
    assert_eq!(shed_of(again(5, &free, Some(&[1]))), gone);
    assert_eq!(picked(&again(1, &free, Some(&[1]))), (1, 102));
}

/// The all-vetoed rule keeps to a side's places. Two prefills, each
/// with its own decode, and the decode list held to the second decode's
/// place, which is over its thresholds. Under a binding that steers the
/// pair is the second prefill with that decode, whatever stands outside
/// the places: a first decode over its thresholds too opens no prefill,
/// and a free one does not keep the pool from the rule. Under a binding
/// that sheds both are refused as saturated. Where the one place holds
/// a decode that is not ready, a decode outside it that is over its
/// thresholds or doubted takes nothing: the rule looks at every row of
/// the side and the pair is refused at the place's own stage, under
/// either word. The gateway's classic router counts only the decodes
/// its request's rule admits when it lifts a decode pool, and pairs
/// among those
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1315-1321`, `:1341`, `:1389-1394`, `:1575-1578`, at main
/// `ebbcd9c6d`).
#[test]
fn the_all_vetoed_rule_keeps_to_a_sides_places() {
    use crate::{document::spec::OverloadSpec, engine::decision::LiftedVeto};
    let spec_of = |shed: bool| {
        let presets = PolicySpec::pair_presets("passthrough", "passthrough");
        let mut document = presets.expect("the pair presets");
        document.generation = 3;
        document.bindings[0].overload = Some(OverloadSpec {
            shed_when_all_vetoed: shed,
            ..OverloadSpec::default()
        });
        AppliedSpec::validate(&document).expect("a valid document")
    };
    let (steers, sheds) = (spec_of(false), spec_of(true));
    let own: Vec<[u32; 1]> = (0..2).map(|at| [at]).collect();
    let classes = PartnerClasses::number(2, own.iter().map(|one| one.as_slice()));
    let free = Constraints::default();
    // The two lists as `health` leaves the decodes, decided with the
    // decode list held to its second place.
    let decide = |spec: &AppliedSpec, health: &dyn Fn(&mut [Candidate])| {
        let (mut prefills, mut decodes) = lists(2, 2);
        stamp(&mut prefills, &mut decodes, &classes, &OPEN);
        health(&mut decodes);
        decide_records(
            spec.engine(),
            &facts(),
            (&free, [unvetoed(None), unvetoed(Some(&[1]))]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        )
    };
    let shed_of = |decision: Decision| match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("not shed: {other:?}"),
    };
    for first_over in [true, false] {
        let over = |decodes: &mut [Candidate]| {
            decodes[0].health.overloaded = first_over;
            decodes[1].health.overloaded = true;
        };
        let steered = decide(&steers, &over);
        assert_eq!(picked(&steered), (2, 102), "first over {first_over}");
        assert_eq!(steered.trace().lifted, LiftedVeto::Overload);
        assert_eq!(
            shed_of(decide(&sheds, &over)),
            (ShedReason::Saturated, "not-overloaded"),
            "first over {first_over}"
        );
    }
    for doubted in [false, true] {
        let outside = |decodes: &mut [Candidate]| {
            decodes[0].health.overloaded = !doubted;
            decodes[0].health.stalled = doubted;
            decodes[1].health.healthy = false;
        };
        for spec in [&steers, &sheds] {
            assert_eq!(
                shed_of(decide(spec, &outside)),
                (ShedReason::NoEndpoints, "healthy"),
                "doubted {doubted}"
            );
        }
    }
}

/// The lifted rule opens no prefill for a decode outside the decode
/// places. Six prefills, each with its own decode, every decode over
/// its thresholds, and the decode list held to the sixth decode's
/// place. Under a binding that steers the pair is the sixth prefill
/// with that decode at every turn, the overload veto lifted. Were the
/// other five decodes to open their prefills too, the prefills' pick,
/// which names four of the six a turn, would miss the sixth on some
/// turns and no pair would be found. The gateway's classic router lifts
/// a decode pool over the workers its request's rule admits, and takes
/// a prefill as a candidate only where one of its partners is open
/// among those
/// (`model_gateway/src/routers/common/placement.rs:1301-1306`,
/// `:1315-1321`, `:1341-1345`, `:1389-1394`, `:1423-1437`, at main
/// `9f303447f`).
#[test]
fn the_lifted_rule_opens_no_prefill_for_a_decode_outside_the_decode_places() {
    use crate::{document::spec::OverloadSpec, engine::decision::LiftedVeto};
    let presets = PolicySpec::pair_presets("round_robin", "round_robin");
    let mut document = presets.expect("the pair presets");
    document.generation = 3;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: false,
        ..OverloadSpec::default()
    });
    let steers = AppliedSpec::validate(&document).expect("a valid document");
    let (mut prefills, mut decodes) = lists(6, 6);
    let own: Vec<[u32; 1]> = (0..6).map(|at| [at]).collect();
    let classes = PartnerClasses::number(6, own.iter().map(|one| one.as_slice()));
    stamp(&mut prefills, &mut decodes, &classes, &OPEN);
    for decode in &mut decodes {
        decode.health.overloaded = true;
    }
    let free = Constraints::default();
    for seed in 0..24 {
        let facts = RequestFacts { seed, ..facts() };
        let decision = decide_records(
            steers.engine(),
            &facts,
            (&free, [unvetoed(None), unvetoed(Some(&[5]))]),
            &pool(&prefills),
            pools(&prefills, &decodes, &classes),
            &mut Scratch::default(),
        );
        assert_eq!(picked(&decision), (6, 106), "seed {seed}");
        assert_eq!(decision.trace().lifted, LiftedVeto::Overload, "seed {seed}");
    }
}

/// The later call picks a decode behind one prefill.
/// Its filter scope opens once with a pool size of two and two decode rows.
#[test]
fn a_decode_call_shows_its_profile_the_decode_pools_facts() {
    use crate::engine::framework::DecisionScope;

    #[derive(Debug)]
    struct ObservePool(Arc<Mutex<Vec<(u32, usize)>>>);

    impl Filter for ObservePool {
        fn name(&self) -> &'static str {
            "observe-pool"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    impl DecisionScope for ObservePool {
        fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], pool: &PoolFacts<'_>) {
            self.0.lock().push((pool.pool_size, rows.len()));
        }

        fn close(&self) {}
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let opens = Arc::clone(&seen);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("observe-pool", move || {
            Box::new(ObservePool(Arc::clone(&opens)))
        })
        .expect("a free filter name");
    let mut document =
        PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
    document
        .profiles
        .get_mut("decode")
        .expect("the decode profile")
        .filter
        .push("observe-pool".into());
    let spec = AppliedSpec::validate_in(&document, &catalog, &[]).expect("a valid document");
    let host = Host::under((spec, Arc::default()), 1, 2);
    let classes = every(1, 2);
    let asks = host.asks(&classes, true);
    let now = Instant::now();
    let mut guard = DecisionGuard::decide_under(&host.engine, Default::default(), &asks, now)
        .expect("a pair decision");
    guard.dispatched(now);
    seen.lock().clear();

    let later = guard
        .decide_decode_under(&asks, now)
        .expect("a decode decision");
    assert!(later.pick().is_some(), "{later:?}");
    assert!(!guard.decode_chosen(), "the later call issued its pick");
    assert_eq!(seen.lock().as_slice(), [(2, 2)]);
}

/// A shed at the later call leaves the original decode chosen.
/// Only the prefill has a booking until dispatch books the decode.
#[test]
fn a_shed_at_the_call_leaves_the_side_chosen() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let mut asks = host.asks(&classes, true);
    let now = Instant::now();
    let local = Default::default();
    let mut guard =
        DecisionGuard::decide_under(&host.engine, local, &asks, now).expect("a pair decision");
    let original = *guard.decision();
    let target = first_target(&original);
    let decode = target.pair.expect("the chosen decode").decode_id;
    let before = *guard.lease().expect("the prefill lease");
    guard.dispatched(now);

    let call = now + Duration::from_millis(1);
    for (hint, reason) in [
        (0, ShedReason::NoEndpoints),
        (1, ShedReason::PolicyDeclined),
    ] {
        asks.0.constraints.subset_hint = Some(hint);

        // The raw answer exposes the shed and its lease field.
        let held = HeldDecision {
            decision: guard.decision(),
            lease: &before.lease,
            index: 0,
            target: guard.target(),
            decode: HeldDecode {
                session: decode_session(&asks.0.facts),
                tokens: before.lease.tokens,
                chars: before.lease.chars,
                bytes: before.lease.bytes,
                chosen: guard.decode_chosen(),
            },
        };
        let answer = host
            .engine
            .decide_decode_under(local, &asks, &held, call)
            .expect("a decode decision");
        assert_eq!(
            shed_for(&answer.decision),
            (reason, "subset-hint"),
            "hint {hint}"
        );
        assert!(answer.lease.is_none(), "hint {hint}");

        // The guard returns the pick it still holds.
        let standing = guard
            .decide_decode_under(&asks, call)
            .expect("the held pick");
        assert_eq!(standing, &original);
        assert_eq!(guard.target(), &target);
        assert!(guard.decode_chosen());
    }

    let after = guard.lease().expect("the held prefill lease");
    assert_eq!((after.lease, after.due), (before.lease, before.due));
    let at = before.lease.issued_micros;
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Prefill, target.id, at),
            ("dispatched", Side::Prefill, target.id, at),
        ]
    );

    guard.dispatched_decode(call);
    assert!(!guard.decode_chosen());
    assert_eq!(
        host.words(),
        [
            ("issued", Side::Decode, decode, at + 1_000),
            ("dispatched", Side::Decode, decode, at),
        ]
    );
}

/// The later decode call makes its request from the document it decides under.
/// It issues a decode lease and records that document's generation in the trace.
#[test]
fn a_decode_calls_request_is_made_from_the_document_the_call_decides_under() {
    struct FromDocument<'a> {
        asks: Asks<'a>,
        document: &'a AppliedSpec,
    }

    impl RequestMaker for FromDocument<'_> {
        fn model(&self) -> &str {
            self.asks.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            assert!(
                std::ptr::eq(spec, self.document),
                "the maker receives the deciding document"
            );
            let mut request = self.asks.make(spec);
            request.facts.seed = spec.generation();
            request
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.asks.workers()
        }
    }

    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let asks = host.asks(&classes, true);
    let first = leased(&host, &asks, Instant::now());
    let target = first_target(&first.decision);
    let lease = first.lease.expect("the prefill lease").lease;
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    let local = Default::default();
    host.engine
        .say(local, &held, dispatched(Side::Prefill, 0), Instant::now());

    let maker = FromDocument {
        asks,
        document: host.engine.standing_spec(),
    };
    let answer = host
        .engine
        .decide_decode_under(local, &maker, &held, Instant::now())
        .expect("a decode decision");
    assert_eq!(
        answer.lease.expect("a decode lease").lease.side,
        Side::Decode
    );
    let trace = answer.decision.trace();
    assert_eq!(
        trace.seed,
        maker.document.generation(),
        "the call uses the seed its maker set"
    );
    assert_eq!(
        trace.generation, trace.seed,
        "the trace names the generation handed to the maker"
    );
}

/// A pair's decode asked again behind a new document is decided under
/// that document, and its side is leased as the pair's first lease
/// states it. No stage hears the issue: the pair's lease is the first
/// document's, and no later word of it reaches a stage, so a stage
/// that heard the issue would never hear the side end.
#[test]
fn a_decode_asked_again_behind_a_new_document_tells_no_stage_its_issue() {
    let host = Host::of(1, 1);
    let classes = every(1, 1);
    let asks = host.asks(&classes, true);
    let first = leased(&host, &asks, Instant::now());
    let target = first_target(&first.decision);
    let lease = first.lease.expect("the prefill lease").lease;
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    host.words();
    let (next, later) = hearing_under(4, None);
    let applied = host.engine.apply(next);
    applied.expect("a number above the standing one");

    let local = Default::default();
    let again = host
        .engine
        .decide_decode_under(local, &asks, &held, Instant::now());
    let again = again.expect("a decode decision");
    let side = again.lease.expect("the side's lease all the same").lease;
    assert_eq!((side.side, side.generation), (Side::Decode, 3));
    let decided_under = again.decision.trace().generation;
    assert_eq!(decided_under, 4, "the document in force decides");
    assert!(host.words().is_empty(), "the first document hears nothing");
    assert!(later.lock().is_empty(), "no stage is told the issue");
}

/// A pair's decode asked again behind a later document that kept the
/// first one's stages tells them the side's issue: they heard the
/// pair's lease issued, and they hear the side's dispatch. A stage
/// hears a lease's issue where it hears that lease's later words, and
/// nowhere else.
#[test]
fn a_decode_asked_again_behind_a_document_that_kept_its_stages_tells_them_its_issue() {
    let ([first, next], heard) = hearing_and_its_next();
    let host = Host::under((first, heard), 1, 1);
    let classes = every(1, 1);
    let asks = host.asks(&classes, true);
    let first = leased(&host, &asks, Instant::now());
    let target = first_target(&first.decision);
    let lease = first.lease.expect("the prefill lease").lease;
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: false,
        },
    };
    host.words();
    let applied = host.engine.apply(next);
    applied.expect("a number above the standing one");

    let local = Default::default();
    let now = Instant::now();
    let again = host.engine.decide_decode_under(local, &asks, &held, now);
    let again = again.expect("a decode decision");
    let side = again.lease.expect("the side's lease").lease;
    assert_eq!((side.side, side.generation), (Side::Decode, 3));
    let decided_under = again.decision.trace().generation;
    assert_eq!(decided_under, 4, "the document in force decides");
    host.engine
        .say(local, &held, dispatched(Side::Decode, 0), now);
    let told: Vec<_> = host.words().iter().map(|word| (word.0, word.1)).collect();
    let both = [("issued", Side::Decode), ("dispatched", Side::Decode)];
    assert_eq!(told, both, "the stages kept hear the side whole");
}

/// A pair's dispatch said behind a later document, for a pair the
/// document before placed, is counted by the counts of sends in force.
/// The later document built its counts new: a send is a fact of its
/// worker, whatever document decided it.
#[test]
fn a_pairs_older_send_is_counted_by_the_counts_in_force() {
    // The stock pair document whose sides count their sends. Under a
    // later number its filters stand in another order: built new.
    let counting = |number: u64| {
        let document = PolicySpec::pair_presets("least_load", "least_load");
        let mut document = document.expect("the pair presets");
        document.generation = number;
        if number > 3 {
            for profile in document.profiles.values_mut() {
                profile.filter.reverse();
            }
        }
        AppliedSpec::validate(&document).expect("a valid document")
    };
    let host = Host::under((counting(3), Arc::default()), 1, 1);
    let classes = every(1, 1);
    let asks = host.asks(&classes, false);
    let now = Instant::now();
    let first = leased(&host, &asks, now);
    let target = first_target(&first.decision);
    let lease = first.lease.expect("the prefill lease").lease;
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: false,
        },
    };
    let applied = host.engine.apply(counting(4));
    applied.expect("a number above the standing one");

    // The prefill's sends, as each count in force reads them.
    let counted = || -> Vec<u32> {
        let row = Candidate {
            id: target.id,
            dp_rank: target.dp_rank,
            ..Default::default()
        };
        let facts = RequestFacts::default();
        let counts = &host.engine.standing_spec().engine().sent_credits;
        let counts = counts.iter();
        counts
            .map(|count| count.read(&facts, &row).requests)
            .collect()
    };
    assert_eq!(counted(), [0], "not said yet");
    let local = Default::default();
    host.engine
        .say(local, &held, dispatched(Side::Prefill, 0), now);
    assert_eq!(counted(), [1], "the count built new takes it");
}

/// A caller held to the decodes it reaches (`decode_places`) is paired
/// by the table of the full lists. The first prefill, a partner of
/// every decode, takes each decode its caller reaches and no other.
/// The second, a partner of the one decode both callers reach, takes
/// that one. The decode records are written whole, each with its class.
#[test]
fn each_gateway_pairs_its_reachable_decodes_by_the_full_lists_table() {
    let all = [0_u32, 1, 2, 3, 4, 5, 6, 7, 8];
    let shared = [1];
    let host = Host::of(2, all.len());
    let classes = PartnerClasses::number(all.len(), [&all[..], &shared[..]]);
    let full_records: Vec<_> = host
        .decodes
        .iter()
        .map(|worker| worker_id(worker.as_ref()))
        .zip(classes.decodes().iter().copied())
        .collect();
    let now = Instant::now();

    let partners = |prefill: usize, places: Option<&[u32]>| {
        let other = worker_id(host.prefills[1 - prefill].as_ref());
        let mut excluded = vec![other];
        let mut drawn = Vec::new();
        loop {
            let mut ask = host.asks(&classes, false);
            ask.1.decode_places = places;
            ask.0.constraints.excluded = &excluded;
            let (decision, _) = host
                .engine
                .decide_without_lease(&ask, now)
                .expect("a pair decision");
            let (chosen, decode) = host.picked(&decision);
            assert_eq!(chosen, prefill, "decode places {places:?}");

            // A soft exclusion allows a repeat once every partner has been drawn.
            let id = full_records[decode].0;
            if excluded.contains(&id) {
                break;
            }
            excluded.push(id);
            drawn.push(decode as u32);
        }

        BUFFERS.with_borrow(|buffers| {
            let written: Vec<_> = buffers.rows[1]
                .iter()
                .map(|row| (row.id, row.pair_class))
                .collect();
            assert_eq!(written, full_records, "decode classes for {places:?}");
        });
        drawn.sort_unstable();
        drawn
    };

    assert_eq!(partners(0, None), all);
    assert_eq!(partners(1, None), shared);

    // Place 1 is shared. Each other place is local to one gateway.
    let gateway_a = [0, 1, 3, 5, 7];
    let gateway_b = [1, 2, 4, 6, 8];
    for places in [gateway_a, gateway_b] {
        assert_eq!(partners(0, Some(&places)), places, "reachable {places:?}");
        assert_eq!(partners(1, Some(&places)), shared, "reachable {places:?}");
    }
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

/// A pair's retry leaves every rank of the engines it tried, a side at
/// a time: a tried prefill takes its engine's other rank out of the
/// prefills and leaves the decodes as they were, a tried decode does
/// the same on its side, and both together move both sides. Named the
/// tried rank alone, each side kept its engine's other rank. That is
/// the gateway's classic answer for a pair: each side steps aside from
/// the engines the request tried
/// (`model_gateway/src/routers/common/placement.rs:1486-1493` and
/// `:1562-1569` at main `75d279287`). The list such a retry is decided by
/// adds each side's ranks in the order of the pair's lists.
#[test]
fn a_pairs_retry_leaves_every_rank_of_the_engines_it_tried() {
    use std::collections::BTreeSet;

    // On each side two ranks of one engine, then two of another.
    let (prefills, prefill_engine) = ranked("prefill", WorkerType::Prefill, &[2, 2]);
    let (decodes, decode_engine) = ranked("decode", WorkerType::Decode, &[2, 2]);
    let classes = every(prefills.len(), decodes.len() as u32);
    let tried_prefill = worker_id(prefills[1].as_ref());
    let tried_decode = worker_id(decodes[0].as_ref());
    let host = Host::over(hearing(), prefills, decodes);
    // The engines a retry's targets stand on, a side each, over the
    // turns of both lists.
    let landed = |tried: &[WorkerId]| {
        let (mut on_prefills, mut on_decodes) = (BTreeSet::new(), BTreeSet::new());
        for seed in 0..32 {
            let mut asks = host.asks(&classes, false);
            asks.0.facts.seed = seed;
            asks.0.facts.attempt = 1;
            asks.0.facts.tried = tried;
            let outcome = leased(&host, &asks, Instant::now());
            for target in named(&outcome.decision) {
                let pair = target.pair.expect("a pair");
                on_prefills.insert(prefill_engine[target.position as usize]);
                on_decodes.insert(decode_engine[pair.decode_position as usize]);
            }
        }
        (on_prefills, on_decodes)
    };
    let (both, second) = (BTreeSet::from([0, 1]), BTreeSet::from([1]));
    assert_eq!(landed(&[]), (both.clone(), both.clone()), "nobody tried");
    assert_eq!(
        landed(&[tried_prefill]),
        (second.clone(), both.clone()),
        "the tried prefill's engine steps aside, and no decode does"
    );
    assert_eq!(
        landed(&[tried_decode]),
        (both, second.clone()),
        "the tried decode's engine steps aside, and no prefill does"
    );
    assert_eq!(
        landed(&[tried_prefill, tried_decode]),
        (second.clone(), second),
        "a tried pair moves on both sides"
    );

    // The list such a retry is decided by, in the order a stage reads it:
    // the tried workers as the host named them, the prefills' other ranks
    // behind them, and the decodes' behind those.
    let prefill = host.engine.pool_members(&host.prefills);
    let decode = host.engine.pool_members(&host.decodes);
    let tried = [tried_prefill, tried_decode];
    let whole = tried_engines(host.engine.loaded(), &tried, &[&prefill, &decode]);
    assert_eq!(
        whole.expect("each list holds ranks"),
        [
            tried_prefill,
            tried_decode,
            worker_id(host.prefills[0].as_ref()),
            worker_id(host.decodes[1].as_ref()),
        ]
    );
}

/// The decode asked again behind a held prefill leaves every rank of
/// the engine its retry tried, as the pair's first decision does.
#[test]
fn the_decode_asked_again_leaves_every_rank_of_the_engine_it_tried() {
    use std::collections::BTreeSet;

    let (prefills, _) = ranked("prefill", WorkerType::Prefill, &[1]);
    let (decodes, decode_engine) = ranked("decode", WorkerType::Decode, &[2, 2]);
    let classes = every(prefills.len(), decodes.len() as u32);
    let held = prefill_alone(worker_id(prefills[0].as_ref()));
    // The first engine's second rank was tried.
    let tried = [worker_id(decodes[1].as_ref())];
    let host = Host::over(hearing(), prefills, decodes);
    let landed = |tried: &[WorkerId]| {
        let mut on_decodes = BTreeSet::new();
        for seed in 0..32 {
            let Asks(mut request, lists, _) = host.asks(&classes, true);
            request.facts.seed = seed;
            request.facts.attempt = 1;
            request.facts.tried = tried;
            let now = Instant::now();
            let again = lists.decide(&host.engine, None, now, &mut request, Some(&held));
            for target in named(&again) {
                let pair = target.pair.expect("a pair");
                on_decodes.insert(decode_engine[pair.decode_position as usize]);
            }
        }
        on_decodes
    };
    assert_eq!(landed(&[]), BTreeSet::from([0, 1]), "nobody tried");
    assert_eq!(
        landed(&tried),
        BTreeSet::from([1]),
        "the tried decode's engine"
    );
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

/// Under `cache_aware` on both sides, a prompt sent to a pair goes to
/// that pair again under any draw: each side reads its own list's
/// column of who was sent the prompt, and its pick is written behind
/// it. Prompts nobody was sent are placed by their draws, on more than
/// one pair.
#[test]
fn a_prompt_sent_to_a_pair_goes_to_that_pair_again() {
    let host = host_under("cache_aware", "cache_aware");
    let sent = prompt(&host, 1_000);
    let first = host.picked(&decide_prompt(&host, &sent, 1, None));
    for seed in 2..34 {
        let again = host.picked(&decide_prompt(&host, &sent, seed, None));
        assert_eq!(again, first, "seed {seed}");
    }
    let fresh: BTreeSet<(usize, usize)> = (0..32_u32)
        .map(|at| {
            let unsent = prompt(&host, 100_000 + 1_000 * at);
            host.picked(&decide_prompt(&host, &unsent, u64::from(at), None))
        })
        .collect();
    assert!(fresh.len() > 1, "{fresh:?}");
}

/// The decode asked again behind a held prefill reads the decodes'
/// column: it goes to the decode its prompt was sent to, under any
/// draw.
#[test]
fn the_decode_asked_again_goes_to_the_decode_its_prompt_was_sent_to() {
    let host = host_under("cache_aware", "cache_aware");
    let sent = prompt(&host, 1_000);
    let (prefill, decode) = host.picked(&decide_prompt(&host, &sent, 1, None));
    let held = prefill_alone(worker_id(host.prefills[prefill].as_ref()));
    for seed in 2..34 {
        let again = decide_prompt(&host, &sent, seed, Some(&held));
        assert_eq!(host.picked(&again), (prefill, decode), "seed {seed}");
    }
}

/// The decode asked again writes no prefill's record. Behind a held
/// prefill that is not the one its prompt was sent to, it goes to the
/// prompt's decode, and the prompt's next first decisions still go to
/// the pair it was sent to, under any draw.
#[test]
fn the_decode_asked_again_writes_no_prefill() {
    let host = host_under("cache_aware", "cache_aware");
    let sent = prompt(&host, 1_000);
    let first = host.picked(&decide_prompt(&host, &sent, 1, None));
    let other = (first.0 + 1) % host.prefills.len();
    let held = prefill_alone(worker_id(host.prefills[other].as_ref()));
    for seed in 2..10 {
        let again = decide_prompt(&host, &sent, seed, Some(&held));
        assert_eq!(host.picked(&again), (other, first.1), "seed {seed}");
    }
    for seed in 10..42 {
        let next = host.picked(&decide_prompt(&host, &sent, seed, None));
        assert_eq!(next, first, "seed {seed}");
    }
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

/// A binding of the decode phase alone listed ahead of the pair's takes
/// a decode request asked alone and leaves the pair its decode side.
/// The decodes' list keeps the record that side reads where the first
/// binding reads none, so a prompt's decode stays there too.
#[test]
fn a_decode_binding_listed_ahead_leaves_the_pair_its_record() {
    let pair = PolicySpec::pair_presets("round_robin", "cache_aware");
    let mut document = pair.expect("the pair presets");
    let stock = PolicySpec::preset("round_robin").expect("a preset");
    let mut ahead = stock.bindings[0].clone();
    ahead.matcher.phase = vec![Phase::Decode];
    ahead.profile = Some("ahead".into());
    document.bindings.insert(0, ahead);
    let turns = stock.profiles.into_values().next();
    let turns = turns.expect("the preset's profile");
    document.profiles.insert("ahead".into(), turns);
    the_decode_stays_and_the_prefills_turn(&host_of(document, 4, 4));
}

/// A host's indexes of cache events: one chain index for every model,
/// of blocks of the sixteen tokens [`prompt`] hashes by.
struct Indexes(Arc<ChainIndex>);

impl overlap::CacheIndexes for Indexes {
    fn index(&self, _: &str) -> Option<Arc<dyn overlap::CacheIndex>> {
        Some(Arc::clone(&self.0) as Arc<dyn overlap::CacheIndex>)
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        Some(16)
    }
}

/// The document of [`decodes_with_blocks`]: prefills that take turns
/// and decodes under `cache_aware`, behind a binding of the decode
/// phase alone listed ahead of the pair's, `cache_aware` too. Where
/// `pair_alone`, the pair's decode profile reads sent prefixes alone
/// (`source: predictor`) and the binding ahead reads cache events too.
/// Where not, the two stand the other way round.
fn decode_binding_ahead(pair_alone: bool) -> PolicySpec {
    use crate::document::spec::{AffinitySource, ScoreSpec, ScoreStage};

    let pair = PolicySpec::pair_presets("round_robin", "cache_aware");
    let mut document = pair.expect("the pair presets");
    let stock = PolicySpec::preset("cache_aware").expect("a preset");
    let mut ahead = stock.bindings[0].clone();
    ahead.matcher.phase = vec![Phase::Decode];
    ahead.profile = Some("ahead".into());
    document.bindings.insert(0, ahead);
    let aware = stock.profiles.into_values().next();
    let aware = aware.expect("the preset's profile");
    document.profiles.insert("ahead".into(), aware);
    let alone = if pair_alone { "decode" } else { "ahead" };
    let profile = document.profiles.get_mut(alone).expect("the profile");
    for stage in &mut profile.score {
        if let ScoreSpec::Params(ScoreStage::PrefixAffinity(prefix)) = stage {
            prefix.source = Some(AffinitySource::Predictor);
        }
    }
    document
}

/// A host of four prefills and four decodes under
/// [`decode_binding_ahead`]. The host's index of cache events holds a
/// block of another prompt for every decode. The decodes' pool keeps
/// the word of the binding ahead, which is never the pair's decode
/// profile's here.
fn decodes_with_blocks(pair_alone: bool) -> Host {
    let host = host_of(decode_binding_ahead(pair_alone), 4, 4);
    let spec = host.engine.spec();
    let engine = spec.engine();
    assert_eq!(
        engine.side_reads_sent_alone(MODEL, Phase::Decode),
        pair_alone
    );
    let pools_word = engine.pool_record_for(MODEL, Phase::Decode);
    assert_eq!(pools_word, Some(!pair_alone), "the binding ahead's");
    let index = Arc::new(ChainIndex::new());
    for (at, decode) in host.decodes.iter().enumerate() {
        let number = index.intern_worker(decode.url()).expect("a number");
        let content_hash = ContentHash(at as u64 + 1);
        let blocks: Vec<StoredBlock> = request_prefix_hashes(&[content_hash])
            .into_iter()
            .map(|seq_hash| StoredBlock {
                seq_hash,
                content_hash,
            })
            .collect();
        let stored = index.apply_stored(number, &blocks, None, &mut ChainBlockMap::new());
        stored.expect("the block is stored");
    }
    host.engine.affinity().set_indexes(Arc::new(Indexes(index)));
    host
}

/// A pair's decode side reads the record of sent prefixes by its own
/// profile's word, whichever profile its pool keeps the record for.
/// The index of cache events holds a block for every decode, and a
/// binding of the decode phase alone is listed ahead of the pair's.
/// Where the pair's decode profile reads sent prefixes alone, a
/// prompt's decode stays, at the pair's first decision and at the
/// decode asked again: that side is handed what it sent to a worker
/// with a block, and its pick is written there. Where the binding
/// ahead reads alone and the pair's decode profile reads cache events
/// too, that side reads the index for those workers, as before: a
/// prompt the index holds for none of them takes its draws.
#[test]
fn a_pairs_decode_side_reads_the_record_by_its_own_profiles_word() {
    the_decode_stays_and_the_prefills_turn(&decodes_with_blocks(true));

    let host = decodes_with_blocks(false);
    let sent = prompt(&host, 1_000);
    let decodes: BTreeSet<usize> = (1..33)
        .map(|seed| host.picked(&decide_prompt(&host, &sent, seed, None)).1)
        .collect();
    assert!(decodes.len() > 1, "{decodes:?}");
}

/// A document that turns a side's word of the record is read by the
/// pair's next decision, inside the second a list's table stands. The
/// word is read once a table, and the table is of the list's value,
/// which is made under one applied document and read under it alone.
/// Every decision here is made at one instant, so a table that stood
/// across the apply would be read by each of them. Under the first
/// document the pair's decode profile reads cache events too: the index
/// holds a block for every decode, the side reads no record, and a
/// prompt the index holds for no decode takes its draws. The next
/// document's decode profile reads sent prefixes alone: its first
/// decision writes the prompt's decode, and the prompt stays there.
#[test]
fn a_document_that_turns_a_sides_word_is_read_by_the_next_decision() {
    let host = decodes_with_blocks(false);
    let sent = prompt(&host, 1_000);
    let now = Instant::now();
    let decode = |seed| {
        host.picked(&decide_prompt_at(&host, &sent, seed, None, now))
            .1
    };
    let mut before = BTreeSet::new();
    for seed in 1..33 {
        before.insert(decode(seed));
    }
    assert!(before.len() > 1, "{before:?}");

    let mut next = decode_binding_ahead(true);
    next.generation = 4;
    let next = AppliedSpec::validate(&next).expect("a valid document");
    let applied = host.engine.apply(next);
    applied.expect("a number above the standing one");
    let first = decode(33);
    for seed in 34..66 {
        assert_eq!(decode(seed), first, "seed {seed}");
    }
}

/// Each side's column is its own list's, whatever the lists' sizes:
/// over two prefills and four decodes a prompt sent to a pair goes to
/// that pair again under any draw.
#[test]
fn a_prompt_stays_with_its_pair_over_lists_of_two_sizes() {
    let pair = PolicySpec::pair_presets("cache_aware", "cache_aware");
    let host = host_of(pair.expect("the pair presets"), 2, 4);
    let sent = prompt(&host, 1_000);
    let first = host.picked(&decide_prompt(&host, &sent, 1, None));
    for seed in 2..34 {
        let again = host.picked(&decide_prompt(&host, &sent, seed, None));
        assert_eq!(again, first, "seed {seed}");
    }
}

/// The maker of a host that counts a key's requests by the decisions it
/// holds: the count stands under the session's number.
struct ByNumber<'a>(Asks<'a>);

impl RequestMaker for ByNumber<'_> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.0.make(spec)
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.0.workers()
    }

    fn session_counted_under(&self) -> SessionCountedUnder {
        SessionCountedUnder::SessionNumber
    }
}

/// The key of the capped cases' request, in the clear.
const TALK: &str = "talk-1";

/// A host of `prefills` and `decodes` workers under the pair of the
/// stock sessions presets and a rule with the in-flight cap `cap`,
/// which places a key on the worker with the fewest requests.
fn ruled_host(cap: u16, prefills: usize, decodes: usize) -> Host {
    use crate::document::spec::Assignment;

    let pair = PolicySpec::pair_presets("manual", "manual");
    let mut document = pair.expect("the pair presets");
    let stock = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rule = stock.bindings[0].sessions.clone();
    let ruled = rule.as_mut().expect("the stock rule");
    (ruled.inflight_cap, ruled.assignment) = (cap, Assignment::MinLoad);
    document.bindings[0].sessions = rule;
    host_of(document, prefills, decodes)
}

/// [`ruled_host`] under an in-flight cap of two.
fn capped_host(prefills: usize, decodes: usize) -> Host {
    ruled_host(2, prefills, decodes)
}

/// The name each side of [`TALK`]'s request is counted under where the
/// session's number names the count: the prefill side's, then the
/// decode side's.
fn numbers() -> [String; 2] {
    let facts = RequestFacts {
        session_key: Identity::plain(TALK),
        ..facts()
    };
    let mut digits = [0; Session::TEXT_LEN];
    [facts.session(), decode_session(&facts)].map(|side| side.text(&mut digits).to_owned())
}

/// `count` requests in flight on `worker`, each counted under `name`.
fn counted_on(worker: &Arc<dyn Worker>, name: &str, count: usize) {
    for _ in 0..count {
        worker.increment_load();
        worker.increment_routing_key_load(name);
    }
}

/// Under a capped sessions rule each side of a pair reads the count of
/// its key's requests on a worker's object, under the name the
/// request's maker states for that side's session. A maker that says
/// nothing is of a host whose own guard counts: the key's text names
/// both sides. A maker of a host that counts by the decisions it holds
/// says the session's number, and each side keeps a session of its
/// own: the prefills' count stands under the prefill side's number and
/// the decodes' under the decode side's. A side whose pin carries the
/// cap under its own name hands the key on, and a count under any
/// other name is no count of that side's key.
#[test]
fn a_pairs_sides_read_their_keys_count_under_the_name_the_maker_states() {
    /// The key's first pair is placed and both its sides dispatched,
    /// two requests of the key are counted on each side's worker
    /// under that side's name of `names`, the prefills' first, and
    /// the key's next pair is decided: each side's verdict.
    fn next(
        host: &Host,
        maker: &impl RequestMaker,
        names: [&str; 2],
    ) -> (SessionVerdict, DecodeVerdict) {
        let now = Instant::now();
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, maker, now)
            .expect("a decision");
        let (prefill, decode) = host.picked(guard.decision());
        guard.dispatched(now);
        guard.dispatched_decode(now);
        counted_on(&host.prefills[prefill], names[0], 2);
        counted_on(&host.decodes[decode], names[1], 2);
        let decided = host.engine.decide_under(GatewayId::LOCAL, maker, now);
        let (decided, _) = decided.expect("an answer").whole().expect("a whole one");
        let pick = *decided.decision.pick().expect("a pick");
        let pair = pick.targets.first().and_then(|target| target.pair);
        (pick.session_verdict, pair.expect("a pair").decode_verdict)
    }

    let numbers = numbers();
    let numbers = [numbers[0].as_str(), numbers[1].as_str()];
    assert_ne!(numbers[0], numbers[1], "each side keeps its own session");
    let classes = every(2, 2);
    let decided = |by_number: bool, names: [&str; 2]| {
        let host = capped_host(2, 2);
        let mut asks = host.asks(&classes, false);
        asks.0.facts.session_key = Identity::plain(TALK);
        if by_number {
            next(&host, &ByNumber(asks), names)
        } else {
            next(&host, &asks, names)
        }
    };
    let (handed_on, kept) = (
        (SessionVerdict::Respill, DecodeVerdict::RESPILL),
        (SessionVerdict::Pinned, DecodeVerdict::PINNED),
    );
    assert_eq!(decided(false, [TALK; 2]), handed_on, "a guard's count");
    assert_eq!(decided(false, numbers), kept, "no count of a guard's");
    assert_eq!(decided(true, numbers), handed_on, "the held decisions'");
    assert_eq!(decided(true, [TALK; 2]), kept, "no held decision's count");
    // Each side under its own session's number: the other side's is no
    // count of its key.
    let crossed = [numbers[1], numbers[0]];
    assert_eq!(decided(true, crossed), kept, "the other side's number");
}

/// A decode asked again under a capped rule reads its key's count as
/// its first decision does, under the name the request's maker states.
/// A host that counts the request on its first decode across the call
/// says so, and its guard counts under the request's key: that
/// decode's record is read with one of the key's requests less. One
/// other request of the key there leaves the request on its decode
/// under a cap of two, and two hand it on. A host that says nothing is
/// read as it counts: the request is one of the key's there. A host
/// that counts by the decisions it holds is read under the decode
/// side's number.
#[test]
fn a_decode_asked_again_reads_its_keys_other_requests_on_its_first_decode() {
    /// A pair's request is decided with its decode sent later, and
    /// its prefill dispatched. `others` requests of the key stand on
    /// the decode it chose, under `name`, and the host's guard counts
    /// the request itself there under the key's text. The decode is
    /// asked again: whether it stands, and the side's verdict.
    fn again(by_number: bool, counted: bool, name: &str, others: usize) -> (bool, DecodeVerdict) {
        let (host, classes) = (capped_host(1, 3), every(1, 3));
        let mut asks = host.asks(&classes, true);
        asks.0.facts.session_key = Identity::plain(TALK);
        asks.0.constraints.counted_on_decode = counted;
        let maker = ByNumber(asks);
        let now = Instant::now();
        let decided = if by_number {
            DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &maker, now)
        } else {
            DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &maker.0, now)
        };
        let mut guard = decided.expect("a decision");
        let chosen = host.picked(guard.decision()).1;
        guard.dispatched(now);
        counted_on(&host.decodes[chosen], TALK, 1);
        counted_on(&host.decodes[chosen], name, others);
        let call = now + Duration::from_millis(1);
        let asked = if by_number {
            guard.decide_decode_under(&maker, call)
        } else {
            guard.decide_decode_under(&maker.0, call)
        };
        let asked = *asked.expect("a decision");
        let pair = target_at(&asked, 0).pair.expect("a pair");
        (host.picked(&asked).1 == chosen, pair.decode_verdict)
    }

    let (stays, handed_on) = (
        (true, DecodeVerdict::PINNED),
        (false, DecodeVerdict::RESPILL),
    );
    assert_eq!(again(false, true, TALK, 1), stays, "one other request");
    assert_eq!(again(false, true, TALK, 2), handed_on, "two others");
    assert_eq!(again(false, false, TALK, 1), handed_on, "read as it counts");
    let [_, decode_number] = numbers();
    assert_eq!(
        again(true, false, &decode_number, 2),
        handed_on,
        "by number"
    );
    assert_eq!(
        again(true, false, TALK, 2),
        stays,
        "no held decision's count"
    );
}

/// A pair's decision asks its request's maker where the key's requests
/// are counted under a document that reads the count, once a decision,
/// and under no other: a rule with no cap asks nothing, and neither
/// does a binding with no rule.
#[test]
fn a_pairs_decision_asks_where_a_key_is_counted_under_a_capped_rule_alone() {
    use std::cell::Cell;

    /// A maker that counts how often it is asked.
    struct Asked<'a>(Asks<'a>, Cell<u32>);

    impl RequestMaker for Asked<'_> {
        fn model(&self) -> &str {
            self.0.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            self.0.make(spec)
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.0.workers()
        }

        fn session_counted_under(&self) -> SessionCountedUnder {
            self.1.set(self.1.get() + 1);
            SessionCountedUnder::KeyText
        }
    }

    let classes = every(1, 1);
    let asked = |host: &Host| {
        let mut asks = host.asks(&classes, false);
        asks.0.facts.session_key = Identity::plain(TALK);
        let maker = Asked(asks, Cell::new(0));
        let decided = host
            .engine
            .decide_under(GatewayId::LOCAL, &maker, Instant::now());
        assert!(decided.is_ok(), "a decision");
        maker.1.get()
    };
    assert_eq!(asked(&capped_host(1, 1)), 1, "a capped rule");
    assert_eq!(asked(&ruled_host(u16::MAX, 1, 1)), 0, "a rule with no cap");
    assert_eq!(asked(&Host::of(1, 1)), 0, "no rule");
}

/// A pick of a decode that holds the whole prompt, among however many
/// holders, moves no count of writes left out through the pair's own
/// decision: the decode side's write is handed the decode side's
/// column. Ten decodes are each sent the prompt once. Then, while a
/// reader holds the record, a thread of its own picks the ninth and the
/// tenth and ends: a write it had kept would be left out there, as the
/// write of a prompt no decode was sent is.
#[test]
fn a_pick_of_a_ninth_decode_holder_leaves_nothing_out() {
    let pair = PolicySpec::pair_presets("round_robin", "cache_aware");
    let host = host_of(pair.expect("the pair presets"), 2, 10);
    // One instant for every decision: the list's table is read once.
    let now = Instant::now();
    let decide_among = |sent: &PromptHashes, decode: u32| {
        let (prefills, class) = (host.prefills.len(), [decode]);
        let classes = PartnerClasses::number(host.decodes.len(), (0..prefills).map(|_| &class[..]));
        let Asks(mut request, lists, _) = host.asks(&classes, false);
        request.facts = facts();
        sent.write_to(&mut request.facts);
        let decision = lists.decide(&host.engine, None, now, &mut request, None);
        assert_eq!(host.picked(&decision).1, decode as usize);
    };
    let sent = prompt(&host, 1_000);
    (0..10).for_each(|decode| decide_among(&sent, decode));
    let affinity = host.engine.affinity();
    let left_out_behind = |sent: &PromptHashes, decodes: &[u32]| {
        affinity.while_held(|| {
            thread::scope(|scope| {
                let picks = scope.spawn(|| {
                    for &decode in decodes {
                        decide_among(sent, decode);
                    }
                });
                picks.join().expect("the picks");
            });
        });
        affinity.left_out()
    };
    assert_eq!(left_out_behind(&sent, &[8, 9]), 0);
    // The count is live in this setup: every block of a prompt no
    // decode was sent is still to add, and its write meets the reader.
    assert_eq!(left_out_behind(&prompt(&host, 2_000), &[9]), 1);
}

/// The prefill side's twin of the case above: the prefill side's write
/// is handed the prefill side's column. Ten prefills are each sent the
/// prompt once, each decision held to its one place of the prefill
/// list. Then, while a reader holds the record, a thread of its own
/// picks the ninth and the tenth and ends, and no write is left out.
#[test]
fn a_pick_of_a_ninth_prefill_holder_leaves_nothing_out() {
    let pair = PolicySpec::pair_presets("cache_aware", "round_robin");
    let host = host_of(pair.expect("the pair presets"), 10, 2);
    // One instant for every decision: the list's table is read once.
    let now = Instant::now();
    let classes = every(host.prefills.len(), host.decodes.len() as u32);
    let decide_at = |sent: &PromptHashes, prefill: u32| {
        let place = [prefill];
        let Asks(mut request, mut lists, _) = host.asks(&classes, false);
        lists.prefill_places = Some(&place);
        request.facts = facts();
        sent.write_to(&mut request.facts);
        let decision = lists.decide(&host.engine, None, now, &mut request, None);
        assert_eq!(host.picked(&decision).0, prefill as usize);
    };
    let sent = prompt(&host, 1_000);
    (0..10).for_each(|prefill| decide_at(&sent, prefill));
    let affinity = host.engine.affinity();
    let left_out_behind = |sent: &PromptHashes, prefills: &[u32]| {
        affinity.while_held(|| {
            thread::scope(|scope| {
                let picks = scope.spawn(|| {
                    for &prefill in prefills {
                        decide_at(sent, prefill);
                    }
                });
                picks.join().expect("the picks");
            });
        });
        affinity.left_out()
    };
    assert_eq!(left_out_behind(&sent, &[8, 9]), 0);
    // The count is live here too: a prompt no prefill was sent has
    // every block still to add, and its write meets the reader.
    assert_eq!(left_out_behind(&prompt(&host, 2_000), &[9]), 1);
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

/// Who placed a side, and whether its rule keeps it from the profile.
type SideWord = (PlacedBy, bool);

/// Each side's word of `decision`: the prefill side's, then the decode
/// side's.
fn side_words(decision: &Decision) -> [SideWord; 2] {
    let pick = decision.pick().expect("a pick");
    let decode = first_target(decision).pair.expect("a pair").decode_placing;
    assert_eq!(pick.decode_untold(), decode.untold());
    [
        (pick.placed_by(), pick.untold()),
        (decode.placed_by(), decode.untold()),
    ]
}

/// How a pair's host sends its decode.
#[derive(Clone, Copy, Debug)]
enum Sent {
    /// With its prefill.
    Together,
    /// Behind its prefill, decided again there.
    Again,
    /// Behind its prefill, with no call.
    Later,
}

/// The words a stage hears of a pair's side: each word and its side.
type SideWords = Vec<(&'static str, Side)>;

/// The request of `key` served through its guard as its host serves
/// it: decided, its prefill dispatched, its decode sent as `sent` says
/// and both sides ended. The two places, each side's word as the
/// guard's pick states it behind every call, and what the stages heard.
fn served(host: &Host, key: u64, sent: Sent) -> ((usize, usize), [SideWord; 2], SideWords) {
    let classes = every(host.prefills.len(), host.decodes.len() as u32);
    let mut asks = host.asks(&classes, !matches!(sent, Sent::Together));
    asks.0.facts.session_key = Identity::hashed(key);
    let now = Instant::now();
    let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
    let mut guard = guard.expect("a decision");
    guard.dispatched(now);
    if matches!(sent, Sent::Again) {
        guard.decide_decode_under(&asks, now).expect("a decision");
    }
    guard.dispatched_decode(now);
    let (places, words) = (host.picked(guard.decision()), side_words(guard.decision()));
    let first = first_target(guard.decision());
    let decode = first.pair.expect("a pair").decode_id;
    guard.completed(&outcome(Side::Prefill, first.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    let heard = host.words().into_iter().map(|word| (word.0, word.1));
    (places, words, heard.collect())
}

/// Each side of a pair is told to the stages by its own word, through
/// the engine's entries and a guard's words. A key's first pair is the
/// stages' on both sides. Its next follows both pins and is the
/// rule's on both. With the worker of its prefill pin down, the stages
/// place the prefill beside a decode the rule placed, and the other
/// way round with the worker of its decode pin down. Under the rule's
/// word `untold` a stage hears the issue and the dispatch of a side
/// the stages placed and of no side the rule placed alone, and the end
/// of both. Under its default word a stage hears every word.
///
/// So it is for a decode sent with its prefill, for one decided again
/// behind its prefill, and for one sent there with no call. A decode
/// the stages placed at the pair's first decision stays the stages'
/// where it is decided again: the call follows the claim that decision
/// made for the key, and places nothing. The pick the guard keeps
/// behind the call states the first decision's word of the prefill
/// side.
///
/// A gateway's sticky map tags a key by its role
/// (`model_gateway/src/policies/registry.rs:762-769`) and selects a
/// pair's two roles apart, each once a request
/// (`model_gateway/src/routers/common/placement.rs:3026`, `:3088`): a
/// role whose pin stands under its cap asks its policy nothing
/// (`registry.rs:794-797`), beside a role its policy places.
#[test]
fn a_guard_tells_each_side_of_a_pair_by_the_sides_own_word() {
    let (prefill, decode) = (Side::Prefill, Side::Decode);
    let stages: SideWord = (PlacedBy::Stages, false);
    // What a stage hears of a request whose sides state `words`.
    let heard = |sent: Sent, words: [SideWord; 2]| -> SideWords {
        let told = match sent {
            Sent::Together => [
                ("issued", prefill),
                ("issued", decode),
                ("dispatched", prefill),
                ("dispatched", decode),
            ],
            Sent::Again | Sent::Later => [
                ("issued", prefill),
                ("dispatched", prefill),
                ("issued", decode),
                ("dispatched", decode),
            ],
        };
        let hears = |side: Side| !words[usize::from(side == decode)].1;
        let mut heard: SideWords = told.into_iter().filter(|word| hears(word.1)).collect();
        heard.extend([("completed", prefill), ("completed", decode)]);
        heard
    };
    for kept in [RulePlaced::Untold, RulePlaced::Told] {
        let host = kept_host(Assignment::Delegate, kept);
        let ruled: SideWord = (PlacedBy::Rule, kept == RulePlaced::Untold);
        let downs = [None, Some(prefill), Some(decode)];
        let ways = [Sent::Together, Sent::Again, Sent::Later];
        let cases = downs
            .into_iter()
            .flat_map(|down| ways.map(|sent| (down, sent)));
        for (nth, (down, sent)) in cases.enumerate() {
            let case = format!("{kept:?}, {down:?} down, {sent:?}");
            // A key of the case's own: the stages place its first
            // pair, which pins it on both sides.
            let key = 100 + nth as u64;
            let (pinned, words, told) = served(&host, key, sent);
            assert_eq!(words, [stages; 2], "{case}: a new key");
            assert_eq!(told, heard(sent, words), "{case}: a new key");
            let gone = match down {
                Some(Side::Prefill) => Some(&host.prefills[pinned.0]),
                Some(Side::Decode) => Some(&host.decodes[pinned.1]),
                _ => None,
            };
            if let Some(worker) = gone {
                worker.set_status(WorkerStatus::NotReady);
            }
            let (places, words, told) = served(&host, key, sent);
            if let Some(worker) = gone {
                worker.set_status(WorkerStatus::Ready);
            }
            // The rule follows a pin that stands, and the stages place
            // the side whose pin's worker is down.
            let by = |side: Side| if down == Some(side) { stages } else { ruled };
            assert_eq!(words, [by(prefill), by(decode)], "{case}");
            assert_eq!(told, heard(sent, words), "{case}");
            let stayed = (places.0 == pinned.0, places.1 == pinned.1);
            assert_eq!(
                stayed,
                (down != Some(prefill), down != Some(decode)),
                "{case}"
            );
        }
    }
}

/// The session table hears the dispatch of a side its rule keeps from
/// the profile. Under a rule whose own assignment places a new key,
/// both sides of the key's first pair are the rule's, and no stage
/// hears their issue or their dispatch. The host takes the pick's
/// second target, the other decode: the decode's dispatch pins the key
/// there, and the key's next pair follows it.
///
/// A gateway's sticky map keeps the worker a key's request went to,
/// whoever placed it (`model_gateway/src/policies/registry.rs:832-841`).
#[test]
fn the_session_table_hears_the_dispatch_of_a_side_kept_from_the_profile() {
    let host = kept_host(Assignment::MinLoad, RulePlaced::Untold);
    let classes = every(2, 2);
    let mut asks = host.asks(&classes, false);
    asks.0.facts.session_key = Identity::hashed(7);
    let now = Instant::now();
    let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
    let mut guard = guard.expect("a decision");
    let ruled: SideWord = (PlacedBy::Rule, true);
    assert_eq!(side_words(guard.decision()), [ruled; 2]);
    let first = host.picked(guard.decision());
    let taken = guard.take(1).expect("a second target");
    let pair = taken.pair.expect("a pair");
    assert_eq!(taken.position as usize, first.0, "the one prefill");
    assert_ne!(pair.decode_position as usize, first.1, "another decode");
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, taken.id), now);
    guard.completed(&outcome(Side::Decode, pair.decode_id), now);
    let ends = [("completed", Side::Prefill), ("completed", Side::Decode)];
    let heard: SideWords = host.words().into_iter().map(|w| (w.0, w.1)).collect();
    assert_eq!(heard, ends);
    let (places, words, heard) = served(&host, 7, Sent::Together);
    assert_eq!(places, (first.0, pair.decode_position as usize));
    assert_eq!((words, heard), ([ruled; 2], ends.to_vec()));
}

/// A guard's decode call dates the claim it makes by its pair's lease.
/// The pair's first decision claims a decode for the key, and that
/// decode is down at the call, a second later: the call claims the
/// other decode. The guard's release for a refused commit is told under
/// the pair's lease and takes the call's claim back. The key's next
/// pair follows the decode the first decision claimed.
#[test]
fn a_guards_decode_call_claims_under_its_pairs_lease() {
    let mut document = kept_document("round_robin", Assignment::Delegate, RulePlaced::Told);
    let rule = document.bindings[0].sessions.as_mut();
    // The claim made last leads the key's pins while it stands.
    rule.expect("the pair's rule").recovery = Recovery::Latest;
    let host = host_of(document, 2, 2);
    let classes = every(2, 2);
    let mut asks = host.asks(&classes, true);
    asks.0.facts.session_key = Identity::hashed(7);
    let now = Instant::now();
    let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
    let mut guard = guard.expect("a decision");
    let first = host.picked(guard.decision());
    guard.dispatched(now);
    let claimed = &host.decodes[first.1];
    claimed.set_status(WorkerStatus::NotReady);
    let call = now + Duration::from_secs(1);
    let again = *guard.decide_decode_under(&asks, call).expect("a decision");
    claimed.set_status(WorkerStatus::Ready);
    let moved = host.picked(&again);
    assert_eq!(moved.0, first.0, "the held prefill");
    assert_ne!(moved.1, first.1, "the call claims the other decode");
    guard.released(ReleaseReason::CommitFailed, call);
    drop(guard);
    let (next, _, _) = served(&host, 7, Sent::Together);
    assert_eq!(next.1, first.1, "the call's claim left with its release");
}

/// A decode call that follows a pin hands the session table its pair's
/// lease date through each of its two entries. One is a guard's call,
/// and the other issues no lease. The pair's first decision is older than
/// the key's idle time by the call. Another request began the key's new
/// life and was served, and the call follows that pin.
///
/// A word under the pair's lease is then of that life, and a word one
/// microsecond older is not.
///
/// A gateway's sticky map pins a decode where that role's selection
/// picks it (`model_gateway/src/policies/manual.rs:322-368`).
#[test]
fn a_decode_call_that_follows_a_pin_dates_its_keys_life_through_each_entry() {
    for leased in [true, false] {
        let mut document = kept_document("round_robin", Assignment::Delegate, RulePlaced::Told);
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the pair's rule");
        (rule.ttl_s, rule.sweep_s, rule.recovery) = (1, 1, Recovery::Latest);
        let host = host_of(document, 2, 2);
        let classes = every(2, 2);
        let mut asks = host.asks(&classes, true);
        asks.0.facts.session_key = Identity::hashed(7);
        // The pair's first decision, and its prefill's dispatch.
        let now = Instant::now() + Duration::from_secs(1);
        let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
        let mut guard = guard.expect("a decision");
        guard.dispatched(now);
        let issued = guard.lease().expect("the pair's lease").lease.issued_micros;
        assert_ne!(issued, 0, "the lease states its instant");
        // The key idles out. Another request of the key begins its new
        // life and is served.
        let later = now + Duration::from_secs(10);
        let other = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, later);
        let mut other = other.expect("a decision");
        other.dispatched(later);
        other.dispatched_decode(later);
        let served = first_target(other.decision())
            .pair
            .expect("a pair")
            .decode_id;
        // The first request's decode call follows that pin.
        let call = later + Duration::from_secs(1);
        let again = if leased {
            *guard.decide_decode_under(&asks, call).expect("a decision")
        } else {
            let held = first_target(guard.decision());
            let own = host.engine.loaded();
            let again = own.decide_decode_without_lease(&asks, &held, issued, call);
            again.expect("a decision")
        };
        let pair = first_target(&again).pair.expect("a pair");
        assert_eq!(pair.decode_id, served, "the new life's pin");
        let verdict = pair.decode_verdict.on(served);
        assert_eq!(verdict, SessionVerdict::Pinned, "{leased}");
        let own = host.engine.loaded();
        let table = own.document.spec.engine().sessions();
        let (key, elsewhere) = (decode_session(&asks.0.facts), WorkerId(9_000));
        table.dispatched(key, SessionVerdict::Pinned, elsewhere, issued - 1);
        let front = || table.pins(key).map(|pins| pins[0]);
        assert_eq!(front(), Some(served), "{leased}: one before");
        table.dispatched(key, SessionVerdict::Pinned, elsewhere, issued);
        assert_eq!(front(), Some(elsewhere), "{leased}: the lease's date");
    }
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
        date_decode_claim(own, &call(verdict), made, 5_000, lease);
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
    date_decode_claim(own, &call(claimed), || key, 5_000, 10);
    table.dispatched(key, claimed, id(101), 10);
    assert_eq!(table.pins(key), Some([id(101), id(0)]), "with a claim");
}

/// A pair's pick of one prefill (worker 1) with two decodes (workers
/// 101 and 102), as the forming makes it of its two sides' lists: each
/// side placed by its stages (`None`), or by the sessions rule alone
/// under the rule's word.
fn placed(prefill: Option<RulePlaced>, decode: Option<RulePlaced>) -> Decision {
    let side = |kept: Option<RulePlaced>, ids: &[u64]| {
        let mut targets = Targets::new();
        for (at, &id) in ids.iter().enumerate() {
            targets.push(Target {
                position: at as u32,
                id: WorkerId(id),
                ..Target::default()
            });
        }
        if let Some(kept) = kept {
            targets.by_rule(kept);
        }
        targets
    };
    let prefill = side(prefill, &[1]);
    let mut targets = side(decode, &[101, 102]);
    let first = *prefill.first().expect("a prefill");
    targets.pair_under(&first, SessionVerdict::None, prefill.placing());
    Decision::Pick(Pick {
        targets,
        ..Pick::default()
    })
}

/// A count of sends takes the dispatch of a side by the side's own
/// word: of a side the stages placed, and of one the rule placed and
/// tells its profile of, whatever the other side states, and of no
/// side the rule keeps from the profile. The counts are read as the
/// engine prints them. No lease here states a session, so the session
/// table stands.
///
/// A gateway's policy counts a request it placed, and none its sticky
/// map placed (`model_gateway/src/policies/registry.rs:794-797`).
#[test]
fn a_count_of_sends_takes_a_sides_dispatch_by_the_sides_own_word() {
    let document = PolicySpec::pair_presets("least_load", "least_load");
    let document = document.expect("the pair presets");
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine = spec.engine();
    let counts = || format!("{engine:?}");
    let words = [None, Some(RulePlaced::Told), Some(RulePlaced::Untold)];
    for (prefill, decode) in words.into_iter().flat_map(|p| words.map(|d| (p, d))) {
        let decision = placed(prefill, decode);
        let sides = first_sides(&facts(), &decision, LeaseId(9));
        let (lease, _, session) = sides.expect("a pair's lease");
        let taken = first_target(&decision);
        let held = HeldDecision {
            decision: &decision,
            lease: &lease,
            index: 0,
            target: &taken,
            decode: HeldDecode {
                session,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: false,
            },
        };
        for (side, kept) in [(Side::Prefill, prefill), (Side::Decode, decode)] {
            let before = counts();
            tell(engine, &held, dispatched(side, 0));
            let counted = counts() != before;
            let told = kept != Some(RulePlaced::Untold);
            assert_eq!(counted, told, "{prefill:?}, {decode:?}: {side:?}");
        }
    }
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
