//! The cases of the frame decision: rules a decision holds on the served
//! path. One case stands in this file. It asks frames of several kinds
//! over workers that state demand facts. Every other case stands in the
//! file of its subject, which reads this file's helpers:
//!
//! - `single`: a single request's frame, and a frame that stands alone;
//! - `pair`: a pair's frame at its prefill;
//! - `decode_call`: a decode call behind a pair, under no sessions rule;
//! - `sessions`: a pair's frame and a decode call under a sessions rule;
//! - `demands`: a frame that states its request's hard demands;
//! - `encode`: an encode request's frame;
//! - `named`: a frame whose request names its worker.

use std::{
    collections::BTreeSet,
    mem,
    panic::resume_unwind,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::load::state::{one_rank_load, LoadSnapshot};

use super::{
    decide, decide_alone, decide_decode, decide_pair, hands_reports, issue, issue_decode,
    issue_pair, Booked, FrameBuffers,
};
use crate::{
    document::{
        applied::AppliedSpec,
        spec::{Assignment, PolicySpec, SessionsSpec},
    },
    engine::{
        decision::{Decision, ShedReason, Target, Targets, SUBSET_HINT, WIRES},
        decision_engine::{Request, RequestMaker, WorkerLists},
        decision_slot::DecisionSlot,
        framework::{DecisionScope, Filter, Picker, TieBand},
        sticky_sessions::SessionVerdict,
        view::{
            AffinityFacts, AffinityHolder, AffinityRows, Candidate, Constraints, Phase, PoolFacts,
            RequestFacts, Role, Session, WireModes, WorkerId,
        },
        DemandFacts,
    },
    policies::random::Random,
    service::{
        controller::{
            decision_slots::{
                tests::{
                    a_frame_under_the_second_files_watch, fleet, frame_for, holder, preset, ready,
                    taken_at, written_ready, KeepsReports, BUDGET, GATEWAY, MODEL,
                },
                Part,
            },
            frame_request::FrameRequest,
            lifecycle::{
                self,
                tests::{a_dispatched, open},
            },
            wire::{Checked, Lists},
        },
        proto,
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{
        pairing::PairingDescriptor, AtWork, DecodeIssue, DecodeMark, GatewayId, Hand, HealthSink,
        LeaseRecord, LeaseState, LeasesStore, LookOutcome, MemoryStore, PickRecord, Reachability,
        WorkerFacts, WorkerStatus,
    },
    tracking::lease::{
        Dispatch, EventKind, EventSet, Lease, LeaseId, Outcome, ReleaseReason, Side, DISPATCH_TTL,
    },
    worker_host::{tests::stand, Members, Objects},
};

mod decode_call;
mod demands;
mod encode;
mod named;
mod pair;
mod sessions;
mod single;

/// A gateway that is not the cases' own.
const OTHER: GatewayId = GatewayId(12);

/// The model of the cases that file a fleet's leases themselves.
const BOOKED: &str = "model-http";

/// The decision of `frame` as `gateway`'s, received and taken at the two
/// instants, with the decider's `place` at the files of leases where the
/// case lends one: what a deciding thread asks for a single request, over
/// the objects of the rows the case wrote through the holder.
fn served(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    gateway: GatewayId,
    taken: (Instant, Instant),
    place: Option<&AtWork>,
) -> Decision {
    served_booked(store, objects, frame, gateway, taken, place).0
}

/// [`served`] with what the lease of a pick takes of its frame, beside the
/// decision.
fn served_booked(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    gateway: GatewayId,
    taken: (Instant, Instant),
    place: Option<&AtWork>,
) -> (Decision, Booked) {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, gateway, taken, place);
    let mut frames = FrameBuffers::default();
    let own = Part::default();
    // The load monitor published nothing: the request reads no report.
    let loads = Arc::new(LoadSnapshot::default());
    let decided = store.deciding(|world| decide(world, objects, &taken, &mut frames, &loads, &own));
    decided.expect("the engine decides")
}

/// The `i`th ready member of [`BOOKED`] in `store`, on the HTTP wire,
/// written through `objects`.
fn member(store: &MemoryStore, objects: &Objects<Members>, i: usize) -> WorkerId {
    let name = format!("h{i}");
    ready(store, objects, &name, BOOKED, Role::Single, WireModes::HTTP)
}

/// The worker a decision names first.
fn first(decision: &Decision) -> Option<WorkerId> {
    let pick = decision.pick()?;
    pick.targets.first().map(|target| target.id)
}

/// `document` as its store applies it, with the stages of `catalog`.
fn staged(document: &PolicySpec, catalog: &StageRegistry) -> AppliedSpec {
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(document, digest, catalog, &[])
        .expect("the catalog holds every stage the document names")
}

/// `frame` with its request's hard demands stated.
fn demanding(mut frame: proto::DecideRequest, demands: proto::Demands) -> proto::DecideRequest {
    frame.demands = Some(demands);
    frame
}

/// `frame` with a session key that hashes to `key`.
fn keyed(mut frame: proto::DecideRequest, key: u64) -> proto::DecideRequest {
    let facts = frame.facts.as_mut().expect("the case's frame has facts");
    facts.session_key = Some(proto::Identity {
        form: Some(proto::identity::Form::Hash(key)),
    });
    frame
}

/// Files one issued lease of [`GATEWAY`] on `target`, under `id`.
fn file(store: &MemoryStore, id: u64, target: WorkerId, now: Instant) {
    file_of(store, id, target, Session::NONE, now);
}

/// [`file`] for a lease of `session`, which counts for its key.
fn file_of(store: &MemoryStore, id: u64, target: WorkerId, session: Session, now: Instant) {
    let leases: &dyn LeasesStore = store;
    let filed = leases.issue(LeaseRecord {
        lease: Lease {
            id: LeaseId(1 << 63 | id),
            session,
            ..Lease::default()
        },
        gateway: GATEWAY,
        request: id,
        target,
        picked: PickRecord::default(),
        decode: None,
        untold: false,
        term: 1,
        issued_at: now,
        dispatch_deadline: now + DISPATCH_TTL,
        state: LeaseState::ISSUED,
    });
    assert_eq!(filed, Ok(()));
}

/// A store under `spec` with one ready member of [`BOOKED`] a count of
/// `held`, each booked by its count of leases on file, and the holder of
/// the store's worker objects: the store, the holder, the members in the
/// counts' order and the instant the leases were issued at.
fn booked(
    spec: AppliedSpec,
    held: &[u64],
) -> (
    Arc<MemoryStore>,
    Arc<Objects<Members>>,
    Vec<WorkerId>,
    Instant,
) {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store.apply(spec).expect("above the unbound start");
    let members: Vec<WorkerId> = (0..held.len())
        .map(|i| member(&store, &objects, i))
        .collect();
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let targets = members.iter().zip(held);
    let targets = targets.flat_map(|(&member, &held)| (0..held).map(move |_| member));
    for (target, id) in targets.zip(1..) {
        file(&store, id, target, now);
    }
    (store, objects, members, now)
}

/// `held` by the worker of `members` each count is of, by worker id.
fn by_id(members: &[WorkerId], held: &[u32]) -> Counts {
    let mut held: Counts = members.iter().copied().zip(held.to_vec()).collect();
    held.sort_unstable();
    held
}

/// The requests in flight a decision read of each candidate, by worker.
type Counts = Vec<(WorkerId, u32)>;

/// A filter that keeps every worker. As a decision opens it reads each
/// candidate's requests in flight, files the case's leases, and reads the
/// candidates again.
#[derive(Debug)]
struct Late {
    store: Weak<MemoryStore>,
    /// The workers a lease is filed on at the next decision's opening.
    late: Arc<Mutex<Vec<WorkerId>>>,
    /// What each decision read: at its opening, and behind the filing.
    read: Arc<Mutex<Vec<[Counts; 2]>>>,
}

impl Lifecycle for Late {}

impl DecisionScope for Late {
    fn open(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], _: &PoolFacts<'_>) {
        let read = || -> Counts {
            let counts = candidates.iter();
            let mut counts: Counts = counts.map(|row| (row.id, facts.in_flight(row))).collect();
            counts.sort_unstable();
            counts
        };
        let opening = read();
        let store = self.store.upgrade().expect("the case's store");
        let late = mem::take(&mut *self.late.lock());
        let now = Instant::now();
        for (target, id) in late.into_iter().zip(1_000..) {
            file(&store, id, target, now);
        }
        self.read.lock().push([opening, read()]);
    }

    fn close(&self) {}
}

impl Filter for Late {
    fn name(&self) -> &'static str {
        "late"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self)
    }
}

/// What a picker's object did, under the object's number: a pick, or the
/// issue of a lease it heard, with the lease's generation.
type Did = (u32, &'static str, u64);

/// The `random` picker under a number of its own, keeping each pick it
/// makes and each issue it hears.
#[derive(Debug)]
struct Which {
    made: u32,
    did: Arc<Mutex<Vec<Did>>>,
}

impl Lifecycle for Which {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.did
            .lock()
            .push((self.made, "issued", lease.generation));
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Issued])
    }
}

impl Picker for Which {
    fn name(&self) -> &'static str {
        "which"
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
        self.did.lock().push((self.made, "picked", 0));
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The model of the pair's cases.
const PAIRED: &str = "model-paired";

/// The pair plan's document with the stages of `catalog`: a request of the
/// prefill or the decode phase under one pair binding, and a single request
/// beside it. Each side's profile keeps the workers `filter` keeps and
/// draws among them by its own picker, the prefills' first of `picks`.
fn pair_plan(catalog: &StageRegistry, filter: &str, picks: [&str; 2]) -> AppliedSpec {
    let [prefill, decode] = picks;
    let document = format!(
        r#"{{
          "apiVersion": "smg.policy/v1",
          "generation": 1,
          "profiles": {{
            "prefills": {{ "filter": {filter}, "pick": "{prefill}" }},
            "decodes": {{ "filter": {filter}, "pick": "{decode}" }}
          }},
          "bindings": [
            {{ "match": {{ "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
              "profiles": {{ "prefill": "prefills", "decode": "decodes" }} }},
            {{ "profile": "prefills" }}
          ]
        }}"#
    );
    let document: PolicySpec = serde_json::from_str(&document).expect("a document of this build");
    staged(&document, catalog)
}

/// The pair plan's document of this build's own stages: each side by a
/// seeded draw among the workers its health filter keeps.
fn stock_pair_plan() -> AppliedSpec {
    let draws = ["random", "random"];
    pair_plan(&StageRegistry::builtin(), r#"["healthy"]"#, draws)
}

/// A ready member `name` of [`PAIRED`] on `role`'s side, on the HTTP wire,
/// written through `objects`: one `only` alone dials where it names a
/// gateway.
fn side_member(
    store: &MemoryStore,
    objects: &Objects<Members>,
    name: &str,
    role: Role,
    only: Option<GatewayId>,
) -> WorkerId {
    stating_member(store, objects, name, role, only, DemandFacts::default())
}

/// [`side_member`] for a worker that states `stated` for a request's hard
/// demands.
fn stating_member(
    store: &MemoryStore,
    objects: &Objects<Members>,
    name: &str,
    role: Role,
    only: Option<GatewayId>,
    stated: DemandFacts,
) -> WorkerId {
    let key = format!("http://{name}:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role,
        wire_modes: WireModes::HTTP,
        models: vec![PAIRED.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        reachable_from: only.map_or(Reachability::Any, Reachability::Gateway),
        demand_facts: stated,
        ..WorkerFacts::default()
    };
    written_ready(store, objects, facts)
}

/// [`side_member`] for a worker whose row says whether it takes images.
fn seeing_member(
    store: &MemoryStore,
    objects: &Objects<Members>,
    name: &str,
    role: Role,
    vision: bool,
) -> WorkerId {
    let key = format!("http://{name}:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role,
        wire_modes: WireModes::HTTP,
        models: vec![PAIRED.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        vision,
        ..WorkerFacts::default()
    };
    written_ready(store, objects, facts)
}

/// The `n`th frame for [`PAIRED`] at its prefill, with what its gateway
/// states of itself: its wires as the wire carries them, and its hint.
fn at_prefill(n: u8, wires: WireModes, hint: Option<u32>) -> proto::DecideRequest {
    let mut frame = frame_for(PAIRED, n, u64::from(n));
    let facts = frame.facts.as_mut().expect("the case's frame has facts");
    facts.phase = proto::Phase::Prefill.into();
    frame.constraints = Some(proto::Constraints {
        wire_modes: u32::from(wires.bits()),
        subset_hint: hint,
        ..proto::Constraints::default()
    });
    frame
}

/// The pair's decision of `frame` as `gateway`'s at `now`: what a deciding
/// thread asks for a pair at its prefill. The load monitor published
/// nothing: the request reads no report.
fn served_pair(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    gateway: GatewayId,
    now: Instant,
) -> Decision {
    let unread = Arc::new(LoadSnapshot::default());
    served_pair_reading(store, objects, frame, (gateway, now), &unread)
}

/// [`served_pair`] under `loads`, the snapshot the load monitor published.
fn served_pair_reading(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    (gateway, now): (GatewayId, Instant),
    loads: &Arc<LoadSnapshot>,
) -> Decision {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, gateway, (now, now), None);
    let mut frames = FrameBuffers::default();
    let own = Part::default();
    let decided =
        store.deciding(|world| decide_pair(world, objects, &taken, &mut frames, loads, &own));
    decided.expect("the engine decides").0
}

/// One frame of [`GATEWAY`]'s decided and its pick leased under term 1, as
/// a deciding thread does both under one load of the world: the decision,
/// the prefill side's lease, and the decode side's where it was issued.
fn leased_pair(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    now: Instant,
) -> (Decision, Lease, Option<Lease>) {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, GATEWAY, (now, now), None);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    store.deciding(|world| {
        let decided = decide_pair(world, objects, &taken, &mut frames, &loads, &own);
        let (decision, booked) = decided.expect("the engine decides");
        let filed = issue_pair(store, world, &taken, 1, &decision, booked);
        let (prefill, decode) = filed.expect("the store's own term").expect("a pick");
        (decision, prefill, decode)
    })
}

/// The prefill and the decode each target of a pick names, in its order.
fn pairs_of(decision: &Decision) -> Vec<(WorkerId, WorkerId)> {
    let targets = decision
        .pick()
        .into_iter()
        .flat_map(|pick| pick.targets.iter());
    let pairs = targets.filter_map(|target| Some((target.id, target.pair?.decode_id)));
    pairs.collect()
}

/// A shed's reason beside the stage it names, as [`shed_of`] reads it.
type Shed = Option<(ShedReason, &'static str)>;

/// A shed's reason beside the stage it names; `None` for any other answer.
fn shed_of(decision: &Decision) -> Shed {
    match decision {
        Decision::Shed(shed) => Some((shed.reason, shed.emptied_by)),
        Decision::Pick(_) | Decision::Queue(_) => None,
    }
}

/// What each answer of a run named, or the one shed each of them was. An
/// answer that names a worker beside one that is shed fails the case, and
/// so do two sheds that differ in their reason or their stage: a demand
/// holds every answer. The side a shed names is not compared.
fn together<A>(answers: impl Iterator<Item = Result<A, Shed>>) -> Result<Vec<A>, Shed> {
    let (mut picked, mut shed) = (Vec::new(), None);
    for (n, answer) in answers.enumerate() {
        match answer {
            Ok(pick) => picked.push(pick),
            Err(stage) => {
                let same = shed.is_none_or(|before| before == stage);
                assert!(same, "answer {n}: another shed");
                shed = Some(stage);
            }
        }
    }
    let Some(stage) = shed else {
        return Ok(picked);
    };
    assert!(picked.is_empty(), "a pick beside a shed");
    Err(stage)
}

/// The `n`th frame for [`PAIRED`] at `phase` that names no lease and
/// states nothing of its gateway.
fn alone_at(n: u8, phase: proto::Phase) -> proto::DecideRequest {
    let mut frame = frame_for(PAIRED, n, u64::from(n));
    let facts = frame.facts.as_mut().expect("the case's frame has facts");
    facts.phase = phase.into();
    frame
}

/// The decision of `frame` as `gateway`'s at `now`: what a deciding
/// thread asks for a frame that stands alone at a side's phase. The load
/// monitor published nothing. With it goes the decider's place at the
/// files of leases where the case lends one.
fn served_alone(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    (gateway, now): (GatewayId, Instant),
    place: Option<&AtWork>,
) -> Decision {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, gateway, (now, now), place);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    let decided =
        store.deciding(|world| decide_alone(world, objects, &taken, &mut frames, &loads, &own));
    decided.expect("the engine decides").0
}

/// The `random` picker, as a stage that says it may refuse by a report of
/// leases.
#[derive(Debug)]
struct ByReports;

impl Lifecycle for ByReports {}

impl Picker for ByReports {
    fn name(&self) -> &'static str {
        "by-reports"
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

    fn refuses_by_reports(&self) -> bool {
        true
    }
}

/// A decode call's frame for [`PAIRED`]: the `n`th of [`at_prefill`]'s that
/// names `lease`, which its reader takes at the decode phase.
fn at_call(n: u8, lease: LeaseId, wires: WireModes, hint: Option<u32>) -> proto::DecideRequest {
    let mut frame = at_prefill(n, wires, hint);
    frame.decode_of = Some(lease.0);
    frame
}

/// The decision of a decode call's `frame` as `gateway`'s at `now`, behind
/// the pair `held`: what a deciding thread asks for the call once the store
/// answered for its lease. With it go the snapshot the load monitor
/// published, and the decider's place at the files of leases where the
/// case lends one.
///
/// The case states the pair and no date of its lease.
fn served_call(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    (gateway, now): (GatewayId, Instant),
    held: (WorkerId, Option<(WorkerId, u16)>),
    (loads, place): (&Arc<LoadSnapshot>, Option<&AtWork>),
) -> Decision {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, gateway, (now, now), place);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let held = (held.0, held.1, Default::default(), 0);
    let decided = store
        .deciding(|world| decide_decode(world, objects, &taken, held, &mut frames, loads, &own));
    decided.expect("the engine decides").0
}

/// One decode call of [`GATEWAY`]'s decided and its pick issued, as a
/// deciding thread does both under one load of the world: the store's
/// answer for the lease the frame names first, the decision behind it, and
/// for a pick the decode side's lease with its issue. The load monitor
/// published nothing.
fn leased_call(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    now: Instant,
) -> (Decision, Option<(Lease, DecodeIssue)>) {
    let request = Checked::new(frame).expect("a frame the record holds");
    let lease = request.decode_of().expect("a decode call's frame");
    let taken = taken_at(&request, GATEWAY, (now, now), None);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    let held = store.held_pair(GATEWAY, lease);
    let held = held.expect("the lease takes a decode side");
    store.deciding(|world| {
        let decided = decide_decode(world, objects, &taken, held, &mut frames, &loads, &own);
        let (decision, booked) = decided.expect("the engine decides");
        let issued = issue_decode(store, world, &taken, lease, &decision, booked);
        (decision, issued.expect("the side stands chosen"))
    })
}

/// The `random` picker, keeping each lease its engine tells it the issue
/// of.
#[derive(Debug)]
struct Told(Arc<Mutex<Vec<Lease>>>);

impl Lifecycle for Told {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push(*lease);
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Issued])
    }
}

impl Picker for Told {
    fn name(&self) -> &'static str {
        "told"
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

/// The `random` picker, whose issue call panics for a decode side that
/// carries [`FAULTY_TOKENS`], with a payload whose own drop panics.
#[derive(Debug)]
struct Faulty;

/// A panic's payload whose own drop panics.
struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        panic!("the case's own panic, at a payload's drop");
    }
}

/// The count of tokens of the call whose issue [`Faulty`] panics at.
const FAULTY_TOKENS: u32 = 666;

impl Lifecycle for Faulty {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        if lease.side == Side::Decode && lease.tokens == FAULTY_TOKENS {
            resume_unwind(Box::new(Bomb));
        }
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Issued])
    }
}

impl Picker for Faulty {
    fn name(&self) -> &'static str {
        "faulty"
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

/// The `random` picker, keeping each word its engine tells it of a lease's
/// side: the call, the side, and the worker the call names.
#[derive(Debug)]
struct SideWords(Arc<Mutex<Vec<SideWord>>>);

/// One call of a stage about a lease's side: the worker is the one the
/// lease books at an issue or at an end, and the target of a dispatch.
type SideWord = (&'static str, Side, WorkerId);

impl Lifecycle for SideWords {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push(("issued", lease.side, lease.booked));
    }

    fn on_dispatched(&self, _: &Lease, target: &Target, side: Side, _: u32) {
        self.0.lock().push(("dispatched", side, target.id));
    }

    fn on_completed(&self, lease: &Lease, _: &Outcome) {
        self.0.lock().push(("completed", lease.side, lease.booked));
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        self.0.lock().push(("released", lease.side, lease.booked));
    }

    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }
}

impl Picker for SideWords {
    fn name(&self) -> &'static str {
        "side-words"
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

/// The pins the standing engine's table holds for `session`.
fn pins_of(store: &MemoryStore, session: Session) -> Option<[WorkerId; 2]> {
    store.with_snapshot(|world| world.spec().engine().sessions().pins(session))
}

/// A `Dispatched` frame of `lease` on `side`, on its pick's target at
/// `index`.
fn sent(lease: &Lease, side: proto::Side, index: u32) -> proto::Dispatched {
    proto::Dispatched {
        side: side.into(),
        target_index: index,
        ..a_dispatched(lease.id.0)
    }
}

/// The keys the worker objects of `names` count, in the names' order: a
/// key counts once on each worker that carries a request of it.
fn keys_on<const N: usize>(objects: &Objects<Members>, names: [&str; N]) -> [usize; N] {
    names.map(|name| {
        let url = format!("http://{name}:8000");
        let object = objects.registry().get_by_url(&url).expect("its object");
        object.routing_key_load()
    })
}

/// These frames state no demand, so what a row states for one changes no
/// decision. One worker stands in each role. Each states
/// a runtime, a pipeline for media by reference, a way to decide a
/// constraint and a window of sixteen tokens, and every frame's prompt is
/// 4,096 tokens long. A single request's frame, a pair's at its prefill, a
/// decode call, a prefill and a decode frame that stand alone and a frame
/// that names its worker are each answered a pick, and it is the pick they
/// are answered over workers that state nothing.
#[test]
fn a_frame_is_decided_over_rows_that_state_demand_facts_as_over_rows_that_state_none() {
    let now = Instant::now();
    // A store under `plan` with one worker a role, and its prefill.
    let fleet = |plan: AppliedSpec, stated: DemandFacts| {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        store.apply(plan).expect("above the unbound start");
        let roles = [
            ("s0", Role::Single),
            ("p0", Role::Prefill),
            ("d0", Role::Decode),
        ];
        let [_, prefill, _] =
            roles.map(|(name, role)| stating_member(&store, &objects, name, role, None, stated));
        (store, objects, prefill)
    };
    let wide = |mut frame: proto::DecideRequest| {
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        facts.prompt_tokens = 4_096;
        frame
    };
    // Each target of an answer: its worker, and the decode beside it.
    let named = |answer: &Decision| -> Vec<(WorkerId, Option<WorkerId>)> {
        let Decision::Pick(pick) = answer else {
            panic!("no pick: {answer:?}");
        };
        let sides = |target: &Target| (target.id, target.pair.map(|pair| pair.decode_id));
        pick.targets.iter().map(sides).collect()
    };
    let answers = |stated: DemandFacts| {
        let asking = (GATEWAY, now);
        // Under the pair plan: a single request, a pair at its prefill and
        // a decode call behind a prefill that was sent.
        let (store, objects, prefill) = fleet(stock_pair_plan(), stated);
        let frame = wide(frame_for(PAIRED, 1, 1));
        let single = served(&store, &objects, frame, GATEWAY, (now, now), None);
        let frame = wide(at_prefill(2, WireModes::NONE, None));
        let pair = served_pair(&store, &objects, frame, GATEWAY, now);
        let unread = Arc::new(LoadSnapshot::default());
        let frame = wide(at_call(3, LeaseId(1), WireModes::NONE, None));
        let held = (prefill, None);
        let call = served_call(&store, &objects, frame, asking, held, (&unread, None));
        // Under a plan of one profile: a frame that stands alone at each
        // side's phase.
        let (store, objects, _) = fleet(preset("random", 1), stated);
        let [ahead, behind] = [proto::Phase::Prefill, proto::Phase::Decode]
            .map(|phase| served_alone(&store, &objects, wide(alone_at(4, phase)), asking, None));
        // Under the stock hash ring document, whose binding names a target
        // source: a frame that names its worker.
        let (store, objects, _) = fleet(preset("consistent_hashing", 1), stated);
        let mut frame = wide(frame_for(PAIRED, 5, 5));
        frame.named_worker = Some(0);
        let reached = served(&store, &objects, frame, GATEWAY, (now, now), None);
        [single, pair, call, ahead, behind, reached].map(|answer| named(&answer))
    };
    let stated = DemandFacts {
        runtime: 7,
        references: true,
        host_pipeline: true,
        window: 16,
        decoding: 3,
    };
    assert_eq!(answers(stated), answers(DemandFacts::default()));
}
