use std::{
    collections::{BTreeSet, VecDeque},
    io, mem,
    panic::resume_unwind,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc as std_mpsc, Arc, OnceLock, Weak,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::load::state::{one_rank_load, LoadReceiver, LoadSnapshot};
use tokio::{
    sync::{mpsc, watch},
    task::JoinSet,
    time::timeout,
};

use super::{
    decode_call, watched, Clock, DecideStats, Deciding, DecisionCall, DecisionSlots, FiledBeside,
    Hand, Job, Part, Reads, Refused, LOOK_CAP, LOOK_FLOOR, PARTS, TERM_MARK,
};
use crate::{
    document::spec::{self, AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, Targets},
        decision_engine::{Decided, WorkerLists},
        decision_slot::DecisionSlot,
        framework::{DecisionScope, Filter, Picker, Scorer, TieBand},
        view::{
            self, fingerprint, Candidate, Constraints, PoolFacts, RequestFacts, Role, WireModes,
            WorkerId,
        },
    },
    policies::{power_of_two::PowerOfTwoMetric, random::Random},
    service::{
        controller::{
            election::ElectionObservation,
            frame_loads::{self, FrameLoads},
            frame_request::FrameRequest,
            gateways::{
                tests::{GRACE, PATIENT},
                Gateways, Vetoed,
            },
            lifecycle,
            standby::{self, Lag},
            watch_workers::Feed,
            wire::{self, Checked},
            worker_objects::WorkerObjects,
        },
        elector::{Elector, TickOutcome, TERM_CEILING},
        proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome},
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{
        DecodeIssue, DecodeMark, GatewayId, HealthSink, LeaseState, LeasesStore, MembershipChange,
        MemoryStore, Registrant, WorkerFacts, WorkerStatus, WorkersStore,
    },
    tracking::lease::{Dispatch, Lease, LeaseId, ReleaseReason, Said, Side, DISPATCH_TTL},
    worker_host::{tests::stand, Members, Objects},
};

impl DecisionSlots {
    /// What a decider's thread tells at its unwind, told by a case.
    pub(in crate::service::controller) fn tell_unwound(&self) {
        self.told.send_replace(true);
    }
}

mod asked_again;
mod busy;
mod decode;
pub(crate) mod encode;
mod faults;
mod fence;
mod gateway;
mod give_up;
mod leases;
mod marks;
mod measured;
mod named;
mod pair;
mod records;
mod reports;
mod ring;
mod sessions;
mod watches;

/// How long a case waits for a decider before it fails.
pub(crate) const PATIENCE: Duration = Duration::from_secs(20);
pub(crate) const MODEL: &str = "model-a";
pub(crate) const GATEWAY: GatewayId = GatewayId(11);
const OTHER: GatewayId = GatewayId(12);
/// The budget a case's frame gives its decision.
pub(crate) const BUDGET: Duration = Duration::from_secs(5);
/// Past the renew deadline of the bed's election.
const FENCED: Duration = Duration::from_secs(1_801);

/// Waits until what a handed frame changes shows, [`PATIENCE`] at
/// most. The frame's file applies it by its decider, as that leaves
/// its work or at its next look, or by the hand: a case that reads at
/// once reads ahead of a decider still at work, or of its look. A
/// case waits by it too for what a decider shows behind its answer:
/// its watch, or that it left its work.
pub(crate) async fn settled(mut shows: impl FnMut() -> bool) {
    let since = Instant::now();
    while !shows() {
        assert!(
            since.elapsed() < PATIENCE,
            "what the case waits for shows within a decider's look"
        );
        tokio::task::yield_now().await;
    }
}

/// What one read of the case's clock does before it shows the hour.
pub(crate) enum Step {
    /// Shows the taken instant plus this much, from this read on.
    At(Duration),
    /// Tells the case a decider reached this read, then waits for the
    /// case's word.
    Hold {
        reached: std_mpsc::SyncSender<()>,
        go: std_mpsc::Receiver<()>,
    },
    Panic,
    /// Panics with a payload whose own drop panics.
    Bomb,
}

/// A panic's payload whose own drop panics.
struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        panic!("the case's own panic, at a payload's drop");
    }
}

/// The clock of a case: each read takes the script's next step, and a
/// read past the script shows what the last one showed. It keeps the
/// name of every thread that read it.
pub(crate) struct Script {
    /// When the bed's candidate took its lease.
    pub(crate) taken: Instant,
    pub(crate) steps: Mutex<VecDeque<Step>>,
    pub(crate) shown: Mutex<Duration>,
    pub(crate) readers: Mutex<Vec<String>>,
}

impl Script {
    pub(crate) fn read(&self) -> Instant {
        let reader = thread::current().name().unwrap_or_default().to_owned();
        self.readers.lock().push(reader);
        let step = self.steps.lock().pop_front();
        match step {
            Some(Step::At(offset)) => *self.shown.lock() = offset,
            Some(Step::Hold { reached, go }) => {
                reached.send(()).expect("the case waits for this read");
                go.recv_timeout(PATIENCE)
                    .expect("the case lets the read go on");
            }
            Some(Step::Panic) => panic!("the case's own panic, inside a decision"),
            Some(Step::Bomb) => resume_unwind(Box::new(Bomb)),
            None => {}
        }
        self.taken + *self.shown.lock()
    }
}

/// The holder of a case's worker objects over `store`, built as the
/// controller builds its own. A case writes its rows through it.
pub(crate) fn holder(store: &Arc<MemoryStore>) -> Arc<Objects<Members>> {
    WorkerObjects::new(Arc::clone(store)).holder()
}

/// A store and its feed with the holder of its worker objects and a
/// load channel nobody publishes on, as the service's shared state takes
/// the four.
pub(crate) fn followed(
    (store, feed): (Arc<MemoryStore>, Feed),
) -> (Arc<MemoryStore>, Feed, Arc<Objects<Members>>, LoadReceiver) {
    let objects = holder(&store);
    (store, feed, objects, frame_loads::tests::silent())
}

/// `standing` with the rest of a run loop's side where no serve runs,
/// as the service's shared state takes the four: the listeners' close
/// nobody says, the role's word nobody tells, and a lag under the bound
/// of no load poll interval. The two words' senders stand to the
/// process's end, so a stream over the state reads no word and no end
/// of its telling.
pub(crate) fn unserved(
    standing: watch::Receiver<ElectionObservation>,
) -> (
    watch::Receiver<ElectionObservation>,
    watch::Receiver<bool>,
    watch::Receiver<Option<u64>>,
    Arc<Lag>,
) {
    static UNTOLD: OnceLock<(watch::Sender<bool>, watch::Sender<Option<u64>>)> = OnceLock::new();
    let (stop, role) = UNTOLD.get_or_init(|| (watch::Sender::new(false), watch::Sender::new(None)));
    let lag = Arc::new(Lag::new(standby::synced_within(Duration::ZERO)));
    (standing, stop.subscribe(), role.subscribe(), lag)
}

/// A store, a candidate of its election and the deciders over both.
struct Bed {
    elector: Elector,
    standing: watch::Sender<ElectionObservation>,
    store: Arc<MemoryStore>,
    script: Arc<Script>,
    /// What the bed's deciders read, for a decider of the case's own.
    reads: Reads,
    counts: Arc<DecideStats>,
    deciders: DecisionSlots,
    /// The members serving [`MODEL`] in the single role, ready.
    workers: Vec<WorkerId>,
    /// The teller of the load channel the bed's deciders hold: a case
    /// publishes a snapshot as the load monitor does.
    loads: watch::Sender<Arc<LoadSnapshot>>,
    sender: mpsc::Sender<wire::Decided>,
    answers: mpsc::Receiver<wire::Decided>,
}

fn applied(document: &PolicySpec) -> AppliedSpec {
    AppliedSpec::validate(document).expect("the document validates")
}

/// A preset of this build as generation `generation`.
pub(crate) fn preset(name: &str, generation: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset(name).expect("a preset of this build");
    document.generation = generation;
    applied(&document)
}

/// The `random` preset as generation 1 with the case's picker, laid
/// under `name`, in its profile's pick.
fn picking(
    name: &'static str,
    picker: impl Fn() -> Box<dyn Picker> + Send + Sync + 'static,
) -> AppliedSpec {
    let mut catalog = StageRegistry::builtin();
    catalog.picker(name, picker).expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = name.into();
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage")
}

/// The row of `facts`, written through the holder of `store`'s worker
/// objects: the row's id. A new row, and a key's next life, is made
/// ready by its connect signal. A row written again where it stood
/// keeps the status its object stands in.
pub(crate) fn written_ready(
    store: &MemoryStore,
    objects: &Objects<Members>,
    facts: WorkerFacts,
) -> WorkerId {
    let written = objects.upsert(store, facts, Registrant::Control);
    let written = written.expect("the store takes the write");
    let id = written.id();
    store.publish_now();
    if matches!(
        written,
        MembershipChange::Registered { .. } | MembershipChange::Replaced { .. }
    ) {
        let row = objects.read_row(id).expect("the row the write laid");
        objects.connect_row(store, id, row.revision);
    }
    id
}

/// The member `name`, serving `model` in `role` and stating `wires`,
/// ready: written again where it stood, with its status as it was.
pub(crate) fn ready(
    store: &MemoryStore,
    objects: &Objects<Members>,
    name: &str,
    model: &str,
    role: Role,
    wires: WireModes,
) -> WorkerId {
    serving(store, objects, name, &[model], role, wires)
}

/// [`ready`] for a member that serves each of `models`.
fn serving(
    store: &MemoryStore,
    objects: &Objects<Members>,
    name: &str,
    models: &[&str],
    role: Role,
    wires: WireModes,
) -> WorkerId {
    let key = format!("http://{name}:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role,
        wire_modes: wires,
        models: models.iter().map(|&model| model.to_owned()).collect(),
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    written_ready(store, objects, facts)
}

/// Three ready members serving [`MODEL`] in the single role, and one in
/// each other role beside the phase that role serves. None states a
/// wire.
pub(crate) fn fleet(
    store: &MemoryStore,
    objects: &Objects<Members>,
) -> (Vec<WorkerId>, [(proto::Phase, WorkerId); 3]) {
    let member = |name: &str, role: Role| ready(store, objects, name, MODEL, role, WireModes::NONE);
    let singles = ["w0", "w1", "w2"].map(|name| member(name, Role::Single));
    let sides = [
        (proto::Phase::Prefill, member("p0", Role::Prefill)),
        (proto::Phase::Decode, member("d0", Role::Decode)),
        (proto::Phase::Encode, member("e0", Role::Encode)),
    ];
    (singles.to_vec(), sides)
}

/// The fleet under `spec` (the unbound generation with none), a
/// candidate that took its lease at the script's instant under term 1
/// when it `leads`, and `deciders` deciders behind queues of `depth`
/// whose clock runs `steps`.
fn bed(
    spec: Option<AppliedSpec>,
    leads: bool,
    (deciders, depth): (usize, usize),
    steps: impl IntoIterator<Item = Step>,
) -> Bed {
    let store = Arc::new(MemoryStore::new().with_decision_slots(deciders));
    let objects = holder(&store);
    let (workers, _sides) = fleet(&store, &objects);
    if let Some(spec) = spec {
        store.apply(spec).expect("above the unbound start");
    }
    // The bed's lease is its own, handed to nobody: an elector alone,
    // which leads at its take.
    let mut elector =
        Elector::alone("controller-0@10.0.0.7:30100", PATIENT).expect("the settings are in order");
    let taken = Instant::now();
    if leads {
        let tick = elector.tick(taken).expect("the take");
        assert_eq!(tick, TickOutcome::Acquired { term: 1 });
        // The election's report tells the store the term it took: a
        // lease is minted under the store's own term.
        store.set_term(1);
    }
    let script = Arc::new(Script {
        taken,
        steps: Mutex::new(steps.into_iter().collect()),
        shown: Mutex::new(Duration::ZERO),
        readers: Mutex::default(),
    });
    let clock: Clock = {
        let script = Arc::clone(&script);
        Arc::new(move || script.read())
    };
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let (loads, published) = watch::channel(Arc::new(LoadSnapshot::default()));
    let reads = Reads {
        store: Arc::clone(&store),
        objects,
        loads: published,
        fence: elector.fence(),
        standing: stood,
        clock,
        counts: Arc::default(),
    };
    let deciders = DecisionSlots::start(deciders, depth, &reads).expect("the threads start");
    let (sender, answers) = mpsc::channel(64);
    Bed {
        counts: Arc::clone(&reads.counts),
        reads,
        elector,
        standing,
        store,
        script,
        deciders,
        workers,
        loads,
        sender,
        answers,
    }
}

/// A decider of the case's own over the bed's reads, for the case's
/// thread. It is not at work until its case seats it.
fn deciding(bed: &Bed) -> Deciding {
    Deciding {
        reads: bed.reads.clone(),
        frames: Default::default(),
        loads: FrameLoads::of(bed.reads.loads.clone()),
        sample_count: Default::default(),
        faulted: false,
        caught: false,
        working: None,
        name: None,
        index: 0,
        marked: None,
        shared: Arc::default(),
    }
}

/// A decider issues into a file of leases of its own: the deciders do
/// not start over a store made for another count of them.
#[tokio::test]
async fn the_deciders_do_not_start_over_a_store_made_for_another_count() {
    let bed = leading([]);
    assert_eq!(bed.store.decision_slots(), 2);
    let refused = DecisionSlots::start(3, 8, &bed.reads).err();
    let refused = refused.expect("two files of leases, three deciders");
    assert_eq!(refused.kind(), io::ErrorKind::InvalidInput);
}

/// The bed most cases stand on: the `random` preset as generation 1, a
/// leader, two deciders behind queues of eight.
fn leading(steps: impl IntoIterator<Item = Step>) -> Bed {
    bed(Some(preset("random", 1)), true, (2, 8), steps)
}

/// A frame for `model` with the case's budget.
pub(crate) fn frame_for(model: &str, id: u8, seed: u64) -> proto::DecideRequest {
    proto::DecideRequest {
        request_id: vec![id],
        facts: Some(proto::RequestFacts {
            model: model.to_owned(),
            seed,
            budget_micros: u32::try_from(BUDGET.as_micros()).expect("five seconds"),
            ..proto::RequestFacts::default()
        }),
        constraints: None,
        again: false,
        ..Default::default()
    }
}

pub(crate) fn frame(id: u8, seed: u64) -> proto::DecideRequest {
    frame_for(MODEL, id, seed)
}

/// [`frame`] with a prompt of eight hashed blocks, the same for every
/// id and seed.
pub(crate) fn hashed(id: u8, seed: u64) -> proto::DecideRequest {
    let mut frame = frame(id, seed);
    let facts = frame.facts.as_mut().expect("the case's frame has facts");
    facts.block_hashes = (1..=8).collect();
    facts.prompt_tokens = 8 * 16;
    frame
}

/// `frame` with `budget` in place of the case's.
fn budgeted(mut frame: proto::DecideRequest, budget: Duration) -> proto::DecideRequest {
    let facts = frame.facts.as_mut().expect("the case's frame has facts");
    facts.budget_micros = u32::try_from(budget.as_micros()).expect("a budget of seconds");
    frame
}

/// `frame` in `phase`.
fn phased(mut frame: proto::DecideRequest, phase: proto::Phase) -> proto::DecideRequest {
    frame
        .facts
        .as_mut()
        .expect("the case's frame has facts")
        .phase = phase.into();
    frame
}

fn checked(frame: proto::DecideRequest) -> Checked {
    Checked::new(frame).expect("a frame the record holds")
}

/// `request` as a decider takes it for `gateway`, received and taken
/// at the two instants with budget left, with the decider's `place` at
/// the leases' files.
pub(crate) fn taken_at<'a>(
    request: &'a Checked,
    gateway: GatewayId,
    (received, now): (Instant, Instant),
    place: Option<&'a crate::store::AtWork>,
) -> DecisionCall<'a, Checked> {
    let taken = DecisionCall::at(request, gateway, received, now);
    taken.expect("budget is left").placed(place)
}

/// The worker the in-process front names first for `frame`, over the
/// list `objects` lends as it stands: the
/// answer reads what the holder held when it was asked. `None` where
/// the front names no worker.
pub(crate) fn lent_pick(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
) -> Option<WorkerId> {
    let request = checked(frame);
    store.deciding(|world| {
        let mut lists = wire::Lists::default();
        let read = lists.of(&request);
        let generation = objects.registry().model_generation(request.model());
        let list = generation
            .as_deref()
            .map_or(WorkerLists::None, |generation| {
                WorkerLists::Single(objects.lent(generation, Role::Single))
            });
        let maker = FrameRequest::new(&request, &read, BUDGET, list);
        let decided = world.front().decide_without_lease(&maker, Instant::now());
        let (decision, _) = decided.expect("the engine decides");
        decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .map(|target| target.id)
    })
}

impl Bed {
    /// Hands `frame` over as `gateway`'s, received `received` after the
    /// take; why it took no place, if it took none.
    fn submit_at(
        &self,
        frame: proto::DecideRequest,
        gateway: GatewayId,
        received: Duration,
    ) -> Option<Refused> {
        let job = Job {
            request: checked(frame),
            gateway,
            received: self.script.taken + received,
            answers: self.sender.clone(),
        };
        self.deciders.submit(job).map(|(refused, _job)| refused)
    }

    fn submit(&self, frame: proto::DecideRequest) -> Option<Refused> {
        self.submit_at(frame, GATEWAY, Duration::ZERO)
    }

    /// The next answer, within the case's patience, as its stream
    /// would send it.
    async fn answer(&mut self) -> proto::DecideAnswer {
        timeout(PATIENCE, self.answers.recv())
            .await
            .expect("an answer within the case's patience")
            .expect("the bed holds a sender")
            .frame()
    }

    /// `frame`'s answer: handed over and waited for.
    async fn ask(&mut self, frame: proto::DecideRequest) -> proto::DecideAnswer {
        assert_eq!(self.submit(frame), None);
        self.answer().await
    }

    /// The engine's own decision for `frame`, made in the case's thread
    /// by the in-process front over the list the bed's holder lends at
    /// the take, as the wire carries it.
    fn engines(&self, frame: &proto::DecideRequest, term: u64) -> proto::DecideAnswer {
        let request = checked(frame.clone());
        let wire = frame.facts.as_ref().expect("the case's frame has facts");
        let left = Duration::from_micros(u64::from(wire.budget_micros));
        let objects = &self.reads.objects;
        let decision = self.store.deciding(|world| {
            let front = world.front();
            let mut lists = wire::Lists::default();
            let read = lists.of(&request);
            let generation = objects.registry().model_generation(request.model());
            let list = generation
                .as_deref()
                .map_or(WorkerLists::None, |generation| {
                    let lent = objects.lent(generation, Role::Single);
                    WorkerLists::Single(lent)
                });
            let maker = FrameRequest::new(&request, &read, left, list);
            let decided = front.decide_without_lease(&maker, self.script.taken);
            decided.expect("the engine decides").0
        });
        wire::decided(frame.request_id.clone(), term, &decision, None, None)
    }
}

/// The reason of a `Busy` answer.
pub(crate) fn busy(answer: &proto::DecideAnswer) -> Reason {
    match &answer.answer {
        Some(Answer::Busy(busy)) => busy.reason(),
        other => panic!("a Busy: {other:?}"),
    }
}

/// The outcome of a decision's answer.
pub(crate) fn outcome(answer: &proto::DecideAnswer) -> &Outcome {
    match &answer.answer {
        Some(Answer::Decision(proto::Decision {
            outcome: Some(outcome),
            ..
        })) => outcome,
        other => panic!("a decision: {other:?}"),
    }
}

/// The worker a pick names first.
pub(crate) fn picked(answer: &proto::DecideAnswer) -> WorkerId {
    match outcome(answer) {
        Outcome::Pick(pick) => WorkerId(pick.targets.first().expect("a target").id),
        other => panic!("a pick: {other:?}"),
    }
}

/// The one lease a pick's answer carries.
pub(crate) fn lease_of(answer: &proto::DecideAnswer) -> &proto::Lease {
    match &answer.answer {
        Some(Answer::Decision(decision)) => match decision.leases.as_slice() {
            [lease] => lease,
            other => panic!("one lease: {other:?}"),
        },
        other => panic!("a decision: {other:?}"),
    }
}

/// `answer` less the lease it carries: what the engine alone decided.
fn unleased(mut answer: proto::DecideAnswer) -> proto::DecideAnswer {
    if let Some(Answer::Decision(decision)) = &mut answer.answer {
        decision.leases.clear();
    }
    answer
}

/// A shed as the engine declines: its reason beside its stage.
fn declined(reason: proto::ShedReason, stage: &str) -> Outcome {
    Outcome::Shed(proto::Shed {
        reason: reason.into(),
        emptied_by: stage.to_owned(),
        retry_after_secs: 1,
        wait_allowed: false,
    })
}

#[tokio::test]
async fn a_decision_names_a_registered_worker_under_the_leaders_term() {
    let mut bed = leading([]);
    for (id, seed) in (1..=16).zip(100..) {
        let frame = frame(id, seed);
        let answer = bed.ask(frame.clone()).await;
        assert_eq!((answer.request_id.as_slice(), answer.term), (&[id][..], 1));
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        // The answer is the engine's own decision over the list its
        // worker objects lend, with its lease.
        assert_eq!(unleased(answer), bed.engines(&frame, 1));
    }
}

/// A scorer that ranks every worker alike and says that it keeps a
/// number of each load report: no stage of this build says so.
#[derive(Debug)]
pub(crate) struct KeepsReports;

impl Lifecycle for KeepsReports {
    fn keeps_loads(&self) -> bool {
        true
    }
}

impl Scorer for KeepsReports {
    fn name(&self) -> &'static str {
        "keeps-reports"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// A decision reads a worker's status on its object, and the object's
/// leads its record's. A worker a drain marks on its object, and one
/// a verdict sets not ready there, are passed by at the next decision,
/// and one set ready again is picked again.
#[tokio::test]
async fn a_worker_whose_object_leaves_ready_is_passed_by_at_the_next_decision() {
    let mut bed = leading([]);
    let (store, objects) = (Arc::clone(&bed.store), Arc::clone(&bed.reads.objects));
    let set = |worker: WorkerId, status: WorkerStatus| {
        if status == WorkerStatus::Draining {
            let row = WorkersStore::get(&*store, worker).expect("the row");
            assert!(objects.drain_row(&store, worker, row.revision));
        } else {
            stand(&objects, &store, worker, status);
        }
    };
    let (draining, unready, kept) = (bed.workers[0], bed.workers[1], bed.workers[2]);
    let mut named = BTreeSet::new();
    for (id, seed) in (1..=48).zip(1..) {
        named.insert(picked(&bed.ask(frame(id, seed)).await));
    }
    assert_eq!(named, BTreeSet::from([draining, unready, kept]));
    set(draining, WorkerStatus::Draining);
    set(unready, WorkerStatus::NotReady);
    for (id, seed) in (49..=96).zip(1..) {
        let answer = bed.ask(frame(id, seed)).await;
        assert_eq!(picked(&answer), kept, "seed {seed}");
    }
    set(unready, WorkerStatus::Ready);
    let mut named = BTreeSet::new();
    for (id, seed) in (97..=144).zip(1..) {
        named.insert(picked(&bed.ask(frame(id, seed)).await));
    }
    assert_eq!(named, BTreeSet::from([unready, kept]));
}

/// Same request, same answer, whichever way the engine is asked. Two
/// stores are laid alike under one document, each with its own engine
/// and its own worker objects. One engine is asked in process, at its
/// entry that issues a lease, over the list its holder lends and the
/// places the case states by hand. The other is asked by frames,
/// through a deciding thread. Under each stock policy every request of
/// a run is answered the same workers in the same order, or the same
/// shed under the same stage: for a gateway that dials every worker,
/// for one whose wires take workers, and for one under a standing veto.
/// Each side counts a request in flight on its worker's object: the
/// store by the lease it files, the host in process as a load guard
/// does.
#[tokio::test]
async fn the_in_process_front_and_the_served_path_decide_alike() {
    const WIRED: GatewayId = GatewayId(21);
    const VETOING: GatewayId = GatewayId(22);
    // A request that states what the stock policies read: a seed, a
    // client key, a session key, a prompt's block hashes and its size.
    let asked = |id: u8| {
        let mut frame = keyed(u64::from(id % 5) + 1, 500 + u64::from(id));
        frame.request_id = vec![id];
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        facts.client_key = u64::from(id % 7).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        facts.block_hashes = (0..8).map(|block| u64::from(id % 3) << 8 | block).collect();
        facts.prompt_tokens = 8 * 16;
        frame
    };
    // The workers an answer names in their order, or its shed's reason
    // and stage.
    let said = |answer: &proto::DecideAnswer| match outcome(answer) {
        Outcome::Pick(pick) => Ok(pick.targets.iter().map(|to| to.id).collect::<Vec<_>>()),
        Outcome::Shed(shed) => Err((shed.reason(), shed.emptied_by.clone())),
        other @ Outcome::Queue(_) => panic!("a pick or a shed: {other:?}"),
    };
    for name in spec::PRESETS {
        let mut bed = bed(Some(preset(name, 1)), true, (1, 8), []);
        let beside = Arc::new(MemoryStore::new().with_decision_slots(1));
        let objects = holder(&beside);
        fleet(&beside, &objects);
        let applied = beside.apply(preset(name, 1));
        applied.expect("above the unbound start");
        // Three more workers on each side, which state their wires: a
        // gateway that states gRPC dials the last two of them alone.
        let both = WireModes::HTTP.union(WireModes::GRPC);
        let stating = [
            ("h0", WireModes::HTTP),
            ("g0", WireModes::GRPC),
            ("b0", both),
        ];
        let mut dialed = Vec::new();
        for (store, objects) in [(&bed.store, &bed.reads.objects), (&beside, &objects)] {
            let more = stating
                .map(|(worker, wires)| ready(store, objects, worker, MODEL, Role::Single, wires));
            dialed = more[1..].to_vec();
        }
        let vetoed = bed.workers[1];
        let until = bed.script.taken + Duration::from_secs(60);
        bed.store.gateway_veto(VETOING, vetoed, Some(until));
        let at = bed.script.taken;
        for id in 1..=30u8 {
            let (gateway, wires) = match id % 3 {
                0 => (GATEWAY, 0),
                1 => (WIRED, 0b010),
                _ => (VETOING, 0),
            };
            let frame = constrained(asked(id), wires, None);
            assert_eq!(bed.submit_at(frame.clone(), gateway, Duration::ZERO), None);
            let served = bed.answer().await;
            let request = checked(frame);
            let made = beside.deciding(|world| {
                let front = world.front();
                let generation = objects.registry().model_generation(MODEL);
                let generation = generation.expect("the fleet serves the model");
                let lent = objects.lent(&generation, Role::Single);
                let members = front.members_of(lent);
                let ids = members.records(front.standing_spec(), front.clock()).ids();
                let place = |worker: &WorkerId| {
                    let at = ids.iter().position(|id| id == worker);
                    u32::try_from(at.expect("a worker of the list")).expect("a short list")
                };
                // What the gateway may take of the list, by hand: the
                // places its wires leave it, and the place its veto
                // holds back, which no rule lifts.
                let mut left: Vec<u32> = dialed.iter().map(place).collect();
                left.sort_unstable();
                let held_back = [place(&vetoed)];
                let (places, stalled): (Option<&[u32]>, &[u32]) = match gateway {
                    WIRED => (Some(left.as_slice()), &[]),
                    VETOING => (None, held_back.as_slice()),
                    _ => (None, &[]),
                };
                let mut lists = wire::Lists::default();
                let read = lists.of(&request);
                let list = WorkerLists::Single(lent);
                let maker = FrameRequest::new(&request, &read, BUDGET, list);
                let maker = maker.held_to(places, stalled, stalled);
                let decision = match front.decide_at(&maker, || at) {
                    Ok(Decided::Whole(decided, _)) => decided.decision,
                    other => panic!("{name}, request {id}: a whole decision: {other:?}"),
                };
                // A host in process counts the request it sends on its
                // worker's object, by a load guard. The store counts a
                // lease on file there.
                let pick = decision.pick();
                if let Some(first) = pick.and_then(|pick| pick.targets.first()) {
                    let sent = usize::try_from(first.position).expect("a place of the list");
                    members.workers()[sent].increment_load();
                }
                decision
            });
            let made = wire::decided(vec![id], 1, &made, None, None);
            assert_eq!(said(&served), said(&made), "{name}, request {id}");
        }
    }
}

/// A served decision writes the record of sent prefixes at its lease's
/// issue, as the door does in process: on workers that publish no cache
/// events a request lands where its prefix was sent.
#[tokio::test]
async fn a_served_request_lands_where_its_prefix_was_sent() {
    let mut bed = bed(Some(preset("cache_aware", 1)), true, (1, 8), []);
    let sent = picked(&bed.ask(hashed(1, 100)).await);
    // Each lease on file books its worker: load alone would send the
    // next request to another.
    for (id, seed) in (2..=4).zip(101..) {
        let returned = picked(&bed.ask(hashed(id, seed)).await);
        assert_eq!(returned, sent, "request {id}");
    }
}

/// An entry of the record of sent prefixes is a prediction made at the
/// decision. A served decision whose last read of the clock finds its
/// budget gone leaves as `Busy` and gives its lease up. Its entry
/// stays, and the next request with that prefix follows it.
#[tokio::test]
async fn a_decision_that_leaves_as_busy_keeps_its_prefix_for_the_next_request() {
    let fresh = |steps: Vec<Step>| bed(Some(preset("cache_aware", 1)), true, (1, 8), steps);
    // Where the late request lands, and a second request that lands
    // elsewhere as the first of its prefix, each on a bed of its own.
    let late = picked(&fresh(Vec::new()).ask(hashed(1, 100)).await);
    let mut elsewhere = None;
    for seed in 101..=132 {
        if picked(&fresh(Vec::new()).ask(hashed(2, seed)).await) != late {
            elsewhere = Some(seed);
            break;
        }
    }
    let seed = elsewhere.expect("a seed that lands on another worker");
    // The same two requests on one bed. The first decision's last read
    // of the clock shows its budget gone, and the second is taken at
    // the first's instant again.
    let steps = [Duration::ZERO, BUDGET, Duration::ZERO].map(Step::At);
    let mut bed = fresh(steps.into());
    let answer = bed.ask(hashed(1, 100)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.store.live(GATEWAY), 0);
    let next = picked(&bed.ask(hashed(2, seed)).await);
    assert_eq!(next, late, "the late decision's entry is followed");
}

/// The workers the engine's record of the prefixes sent for [`MODEL`]
/// holds a place for, in the bed's store: `None` where the engine keeps
/// no record of the model.
fn sent_prefixes(bed: &Bed) -> Option<Vec<WorkerId>> {
    let held = bed.store.deciding(|world| world.front().affinity().held());
    let of_model = |(model, _): &(Box<str>, _)| **model == *MODEL;
    held.into_iter().find(of_model).map(|(_, workers)| workers)
}

/// The bed under the `cache_aware` preset behind one request of a hashed
/// prompt, with the worker the request was sent to and the facts that
/// worker is registered under: the model's record holds a place for
/// that worker alone.
async fn sent_one_prefix() -> (Bed, WorkerId, WorkerFacts) {
    let mut bed = bed(Some(preset("cache_aware", 1)), true, (1, 8), []);
    let sent = picked(&bed.ask(hashed(1, 100)).await);
    assert_eq!(sent_prefixes(&bed), Some(vec![sent]));
    let row = WorkersStore::get(&*bed.store, sent).expect("the row");
    (bed, sent, row.facts)
}

/// The record of sent prefixes stands across an apply: the store keeps
/// one engine for its life, and the record is that engine's. A request
/// whose prefix was sent under the generation before lands where the
/// prefix was sent.
#[tokio::test]
async fn a_request_lands_where_its_prefix_was_sent_under_the_generation_before() {
    let (mut bed, sent, _) = sent_one_prefix().await;
    let next = preset("cache_aware", 2);
    bed.store.apply(next).expect("a later generation");
    assert_eq!(sent_prefixes(&bed), Some(vec![sent]));
    // Each lease on file books its worker: load alone would send the
    // next request to another.
    for (id, seed) in (2..=4).zip(101..) {
        let returned = picked(&bed.ask(hashed(id, seed)).await);
        assert_eq!(returned, sent, "request {id}");
    }
}

/// The store tells its one engine each word of its membership, under
/// whichever generation stands. A worker that leaves behind an apply
/// leaves the record of the prefixes it was sent under the generation
/// before, and registered again under its id it is named for none of
/// them.
#[tokio::test]
async fn a_worker_that_left_behind_an_apply_is_named_for_no_prefix_it_held_before() {
    let (bed, sent, facts) = sent_one_prefix().await;
    let next = preset("cache_aware", 2);
    bed.store.apply(next).expect("a later generation");
    assert_eq!(sent_prefixes(&bed), Some(vec![sent]));
    let row = WorkersStore::get(&*bed.store, sent).expect("the row");
    bed.store.remove(sent, row.revision).expect("removed");
    assert_eq!(
        sent_prefixes(&bed),
        Some(Vec::new()),
        "the record forgot it"
    );
    assert_eq!(bed.store.upsert(facts).id(), sent, "its id again");
    assert_eq!(sent_prefixes(&bed), Some(Vec::new()));
}

/// `facts` written through the bed's holder, as a registration call
/// writes a row: the row's id.
fn rewritten(bed: &Bed, facts: WorkerFacts) -> WorkerId {
    let (store, objects) = (&bed.store, &bed.reads.objects);
    let written = objects.upsert(store, facts, Registrant::Control);
    written.expect("the store takes the write").id()
}

/// A served engine's records of sent prefixes hear the store's word of
/// a replacement. A key registered again under a new incarnation is a
/// new id, and the records keep no place for the id that went.
#[tokio::test]
async fn the_sent_prefixes_of_a_replaced_incarnation_are_forgotten() {
    let (bed, sent, facts) = sent_one_prefix().await;
    let next = WorkerFacts {
        incarnation: facts.incarnation + 1,
        ..facts
    };
    let replaced = rewritten(&bed, next);
    assert_ne!(replaced, sent, "a new incarnation is a new id");
    assert_eq!(sent_prefixes(&bed), Some(Vec::new()));
}

/// The records hear the store's word of a worker that stands under its
/// id and serves a model no more. A registration that keeps the
/// worker's models keeps what the worker was sent, whether it changes a
/// limit of the worker or its width. One that takes the model off
/// leaves the records no place for the worker.
#[tokio::test]
async fn the_sent_prefixes_of_a_worker_that_left_its_model_are_forgotten() {
    let (bed, sent, facts) = sent_one_prefix().await;
    let relabelled = WorkerFacts {
        max_running: Some(8),
        ..facts.clone()
    };
    let resized = WorkerFacts {
        dp_size: 2,
        ..facts.clone()
    };
    for kept in [relabelled, resized] {
        assert_eq!(rewritten(&bed, kept), sent);
        assert_eq!(sent_prefixes(&bed), Some(vec![sent]));
    }
    let elsewhere = WorkerFacts {
        models: vec!["model-b".to_owned()],
        ..facts
    };
    assert_eq!(rewritten(&bed, elsewhere), sent, "the id stands");
    assert_eq!(sent_prefixes(&bed), Some(Vec::new()));
}

/// Sixteen requests of one stream in flight at once, and their answers:
/// a pick's lease id, or the reason of a `Busy`.
pub(crate) async fn burst(
    deciders: &DecisionSlots,
    sender: &mpsc::Sender<wire::Decided>,
    answers: &mut mpsc::Receiver<wire::Decided>,
) -> Vec<Result<u64, Reason>> {
    burst_of(deciders, sender, answers, frame).await
}

/// [`burst`] of the frames `framed` makes of each request's id and seed.
pub(crate) async fn burst_of(
    deciders: &DecisionSlots,
    sender: &mpsc::Sender<wire::Decided>,
    answers: &mut mpsc::Receiver<wire::Decided>,
    framed: fn(u8, u64) -> proto::DecideRequest,
) -> Vec<Result<u64, Reason>> {
    for id in 0..16 {
        let job = Job {
            request: checked(framed(id, u64::from(id))),
            gateway: GATEWAY,
            received: Instant::now(),
            answers: sender.clone(),
        };
        assert!(deciders.submit(job).is_none(), "the queue takes sixteen");
    }
    let mut read = Vec::new();
    for _ in 0..16 {
        let answer = timeout(PATIENCE, answers.recv())
            .await
            .expect("an answer within the case's patience")
            .expect("the case holds a sender")
            .frame();
        read.push(match &answer.answer {
            Some(Answer::Decision(decision)) => match decision.leases.as_slice() {
                [lease] => Ok(lease.id),
                other => panic!("one lease a pick: {other:?}"),
            },
            _ => Err(busy(&answer)),
        });
    }
    read
}

/// The word a term's mark says, as a case hears it: each call tells the
/// case, then waits for its go.
fn heard(bed: &Bed) -> (std_mpsc::Receiver<()>, std_mpsc::SyncSender<()>) {
    let (reached, words) = std_mpsc::sync_channel(8);
    let (go, held) = std_mpsc::sync_channel::<()>(8);
    let held = Mutex::new(held);
    bed.deciders.leave_through(move |_term| {
        reached.send(()).expect("the case hears the word");
        held.lock().recv().expect("the case lets the decider go");
    });
    (words, go)
}

/// What the engine told a stage of leases: an issue by its id, a release
/// by its id and reason.
pub(crate) type Calls = Vec<(u64, Option<ReleaseReason>)>;

/// The `random` picker, keeping what the engine tells it of leases.
#[derive(Debug)]
struct Told(Arc<Mutex<Calls>>);

impl Lifecycle for Told {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push((lease.id.0, None));
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.0.lock().push((lease.id.0, Some(reason)));
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

/// The `random` preset as `generation`, picking through [`Told`]: what
/// that generation's engine tells its stage of leases is kept in
/// `told`.
pub(crate) fn hearing(told: &Arc<Mutex<Calls>>, generation: u64) -> AppliedSpec {
    let kept = Arc::clone(told);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("told", move || Box::new(Told(Arc::clone(&kept))))
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = generation;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "told".into();
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage")
}

/// `frame` as its gateway sends it again, holding no answer to the
/// first ask.
fn again(mut frame: proto::DecideRequest) -> proto::DecideRequest {
    frame.again = true;
    frame
}

/// The requests the bed's deciders counted as asked again: those that
/// found no lease waiting, and those whose earlier lease was ended.
fn asked_again(bed: &Bed) -> [u64; 2] {
    let again = bed.counts.again.each_ref();
    again.map(|count| count.load(Ordering::Relaxed))
}

/// [`Seen`], as a stage that says it may refuse by a report of
/// leases.
#[derive(Debug)]
struct Counted(Seen);

impl Lifecycle for Counted {}

impl Picker for Counted {
    fn name(&self) -> &'static str {
        self.0.name()
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
        self.0.pick(facts, candidates, kept, scores, band, out);
    }

    fn refuses_by_reports(&self) -> bool {
        true
    }
}

/// The `random` preset as generation 1 with the case's picker in its
/// profile's pick: a picker that keeps what each decision reads in
/// `rows`, made by `stage`.
fn seeing<P: Picker + 'static>(
    rows: &Arc<Mutex<Vec<SeenRows>>>,
    stage: impl Fn(Seen) -> P + Send + Sync + 'static,
) -> AppliedSpec {
    let kept = Arc::clone(rows);
    picking("seen", move || {
        let rows = Arc::clone(&kept);
        Box::new(stage(Seen { rows }))
    })
}

/// Two leases on the second file of `store`, booked on `member`, under
/// the watch of that file's place, and the release of the first handed
/// and left in the file's ring. The place is the case's own, so no
/// decider's look applies the frame: its next look says whether the
/// frame still waited ([`LookOutcome::Fed`]) or was applied meanwhile
/// ([`LookOutcome::Quiet`]).
pub(crate) fn a_frame_under_the_second_files_watch(
    store: &Arc<MemoryStore>,
    member: WorkerId,
) -> crate::store::AtWork {
    use crate::{
        engine::decision_slot::DecisionSlot,
        store::{Hand, LookOutcome},
        tracking::lease::{ReleaseReason, Said},
    };

    let now = Instant::now();
    let _acting = DecisionSlot::new(1).enter();
    let mut place = store.at_work();
    let issue = |request| {
        let record = crate::store::LeaseRecord {
            lease: Lease::default(),
            gateway: GATEWAY,
            request,
            target: member,
            picked: crate::store::PickRecord::default(),
            decode: None,
            untold: false,
            term: 1,
            issued_at: now,
            dispatch_deadline: now + DISPATCH_TTL,
            state: LeaseState::ISSUED,
        };
        let booked = store.snapshot().booked_of(member);
        let issued = store.issue_decided(record, booked, |_| ());
        issued.expect("minted under the store's term").id
    };
    let first = issue(1);
    issue(2);
    assert_eq!(
        place.look(),
        LookOutcome::Quiet,
        "two leases stand on its file"
    );
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    store.hand(&Hand::new(GATEWAY), first, gone);
    place
}

/// One model a decider, in the deciders' order: the first names of an
/// endless run whose requests each of `count` deciders takes.
fn a_model_each(deciders: &DecisionSlots, count: usize) -> Vec<String> {
    let mut models: Vec<Option<String>> = vec![None; count];
    for name in (0_u32..).map(|at| format!("model-{at}")) {
        if models.iter().all(Option::is_some) {
            break;
        }
        let taker = &mut models[deciders.of(&name)];
        if taker.is_none() {
            *taker = Some(name);
        }
    }
    models.into_iter().flatten().collect()
}

/// What one decision was handed of each row's requests in flight: the
/// count the request's facts read, and the row's own field.
type SeenRows = Vec<(u32, u32)>;

/// The `random` picker, keeping the reads of every decision.
#[derive(Debug)]
struct Seen {
    rows: Arc<Mutex<Vec<SeenRows>>>,
}

impl Lifecycle for Seen {}

impl Picker for Seen {
    fn name(&self) -> &'static str {
        "seen"
    }

    // Keeps what a decision reads and picks as the random picker.
    fn refuses_by_reports(&self) -> bool {
        false
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
        let counts = candidates
            .iter()
            .map(|row| (facts.in_flight(row), row.in_flight));
        self.rows.lock().push(counts.collect());
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The deciders as they stand once their one thread ended: its queue
/// has no reader.
pub(crate) fn gone(reads: &Reads) -> DecisionSlots {
    let (queue, jobs) = std_mpsc::sync_channel(1);
    drop(jobs);
    DecisionSlots {
        queues: Box::new([queue]),
        told: Arc::new(watch::Sender::new(false)),
        reads: reads.clone(),
        shared: Arc::default(),
    }
}

/// A decider's thread that ended takes no request: each one for its
/// models comes back whole, as gone and never as a full queue.
#[test]
fn a_request_for_a_decider_whose_thread_ended_comes_back_as_gone() {
    let deciders = gone(&leading([]).reads);
    let (answers, _unread) = mpsc::channel(1);
    let job = Job {
        request: checked(frame(1, 1)),
        gateway: GATEWAY,
        received: Instant::now(),
        answers,
    };
    let back = deciders
        .submit(job)
        .map(|(refused, job)| (refused, job.request.into_id()));
    assert_eq!(back, Some((Refused::Gone, vec![1])));
}

#[tokio::test]
async fn a_models_requests_are_decided_by_one_decider_in_the_order_they_were_handed_over() {
    let mut bed = bed(Some(preset("random", 1)), true, (4, 64), []);
    for id in 1..=48 {
        assert_eq!(bed.submit(frame(id, u64::from(id))), None);
    }
    for id in 1..=48 {
        assert_eq!(bed.answer().await.request_id, [id]);
    }
    let own = format!("decide-{}", bed.deciders.of(MODEL));
    let readers = mem::take(&mut *bed.script.readers.lock());
    assert_eq!(readers.len(), 96, "two reads a decision");
    assert!(readers.iter().all(|reader| *reader == own), "{readers:?}");
    // A model with another key is another decider's, whoever asks.
    let other = (0..64)
        .map(|index| format!("model-{index}"))
        .find(|model| bed.deciders.of(model) != bed.deciders.of(MODEL))
        .expect("some model of sixty-four has another key");
    for gateway in [GATEWAY, OTHER] {
        let frame = frame_for(&other, 1, 1);
        assert_eq!(bed.submit_at(frame, gateway, Duration::ZERO), None);
        bed.answer().await;
    }
    let theirs = format!("decide-{}", bed.deciders.of(&other));
    let readers = mem::take(&mut *bed.script.readers.lock());
    assert!(
        !readers.is_empty() && readers.iter().all(|reader| *reader == theirs),
        "{readers:?}"
    );
}

/// The bed's one worker of `role`: its fleet holds one a side.
fn side_of(bed: &Bed, role: Role) -> WorkerId {
    let (_, rows) = bed.store.all_rows();
    let row = rows.iter().find(|row| row.facts.role == role);
    row.expect("the fleet holds one worker a side").id
}

/// What the engine told a stage of a lease's sides: an issue and a
/// release, each by the side and the worker its lease books.
type SideCalls = Vec<(Side, WorkerId, Option<ReleaseReason>)>;

/// A pair's one request under `id`, whose decode is asked for after
/// its prefill where `later`.
fn pair_frame(id: u8, later: bool) -> proto::DecideRequest {
    let mut frame = phased(frame(id, 77), proto::Phase::Prefill);
    frame.decode_later = later;
    frame
}

/// The decode call of the pair's lease `lease`, for the request `id`.
fn call(id: u8, lease: u64) -> proto::DecideRequest {
    let mut frame = pair_frame(id, false);
    frame.decode_of = Some(lease);
    frame
}

/// A dispatch of `side` to the pick's first target.
fn sent(side: Side) -> Dispatch {
    Dispatch {
        side,
        target_index: 0,
        since_decision_micros: 0,
    }
}

/// The worker and the decode of each target of a pick's answer, and
/// each lease's id, side and target index.
type PairAnswer = (Vec<(u64, Option<u64>)>, Vec<(u64, proto::Side, u32)>);

fn pair_of(answer: &proto::DecideAnswer) -> PairAnswer {
    let Outcome::Pick(pick) = outcome(answer) else {
        panic!("a pick: {answer:?}");
    };
    let decode = |target: &proto::Target| target.pair.as_ref().map(|pair| pair.decode_id);
    let targets = pick
        .targets
        .iter()
        .map(|target| (target.id, decode(target)));
    (targets.collect(), sides_of(answer))
}

fn sides_of(answer: &proto::DecideAnswer) -> Vec<(u64, proto::Side, u32)> {
    let Some(Answer::Decision(decision)) = &answer.answer else {
        panic!("a decision: {answer:?}");
    };
    let side = |lease: &proto::Lease| (lease.id, lease.side(), lease.target_index);
    decision.leases.iter().map(side).collect()
}

/// What the leases on file book on each of `workers`, as the worker
/// objects count it.
fn booked<const N: usize>(bed: &Bed, workers: [WorkerId; N]) -> [usize; N] {
    workers.map(|id| {
        let row = WorkersStore::get(&*bed.store, id).expect("the row");
        let object = bed.reads.objects.registry().get_by_url(&row.facts.key);
        object.expect("its object").load()
    })
}

/// The decode side of `lease` as the file holds it.
fn decode_on_file(bed: &Bed, lease: u64) -> (LeaseState, WorkerId) {
    let sides = bed.store.sides(GATEWAY, LeaseId(lease));
    let decode = sides.and_then(|sides| sides.decode);
    let decode = decode.expect("the decode side on file");
    (decode.state, decode.target)
}

#[tokio::test]
async fn the_engine_answers_where_no_generation_binds_the_model_and_where_no_member_serves_it() {
    // The unbound generation: number 0, declined by the binding.
    let mut unbound = bed(None, true, (2, 8), []);
    let frame = frame(1, 1);
    let answer = unbound.ask(frame.clone()).await;
    assert_eq!(
        *outcome(&answer),
        declined(proto::ShedReason::PolicyDeclined, "binding")
    );
    assert_eq!(answer, unbound.engines(&frame, 1));
    assert_eq!(unbound.counts.awaited.load(Ordering::Relaxed), 0);
    // A model no member serves: its decider hands the engine no list,
    // and the answer is the engine's own for a host that holds no
    // worker of a model (`single::over_none`): no endpoint, under no
    // stage's name, whatever the document. The unbound generation's
    // binding is not asked.
    let none = declined(proto::ShedReason::NoEndpoints, "none");
    let answer = unbound.ask(frame_for("model-nobody-serves", 3, 3)).await;
    assert_eq!(*outcome(&answer), none);
    let mut bed = leading([]);
    let frame = frame_for("model-nobody-serves", 2, 2);
    let answer = bed.ask(frame.clone()).await;
    assert_eq!(*outcome(&answer), none);
    assert_eq!(answer, bed.engines(&frame, 1));
}

/// While the start's document of a replica under a shared lease waits
/// for its first lead, a request is no decision: `Busy` with no
/// reason, so with no lease and no trace, under a count of its own. The
/// first request behind the document's landing is decided under it.
#[tokio::test]
async fn a_request_ahead_of_the_starts_document_is_no_decision() {
    let mut bed = bed(None, true, (2, 8), []);
    bed.store.snapshot().spec().mark_document_awaited();
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!(
        (busy(&answer), answer.request_id.as_slice(), answer.term),
        (Reason::Unspecified, &[1][..], 1)
    );
    let counted = |count: &AtomicU64| count.load(Ordering::Relaxed);
    assert_eq!(
        (
            counted(&bed.counts.awaited),
            counted(&bed.counts.pair_sides)
        ),
        (1, 0)
    );
    // The document lands, and the mark goes with the unbound start.
    bed.store
        .apply(preset("random", 1))
        .expect("above the unbound start");
    let answer = bed.ask(frame(2, 1)).await;
    assert!(matches!(outcome(&answer), Outcome::Pick(_)), "{answer:?}");
    assert_eq!(counted(&bed.counts.awaited), 1);
}

/// `frame` with what its gateway alone knows.
fn constrained(
    mut frame: proto::DecideRequest,
    wires: u32,
    hint: Option<u32>,
) -> proto::DecideRequest {
    frame.constraints = Some(proto::Constraints {
        wire_modes: wires,
        subset_hint: hint,
        ..proto::Constraints::default()
    });
    frame
}

/// The counts a made decision can move are kept in parts, a decider's
/// own at its index, so no line is written by two of them, and each is
/// read whole. A model on each of the bed's two deciders, each with one
/// member its gateway cannot dial, and each decider's first request
/// with one block hash more than the view reads: a decision counts in
/// its own decider's part, and deciders past the parts share one.
#[tokio::test]
async fn a_decider_counts_what_its_decisions_move_in_its_own_part() {
    let mut bed = leading([]);
    let store = Arc::clone(&bed.store);
    let part = |counts: &DecideStats, index| {
        let part = counts.part(index);
        let read = |count: &AtomicU64| count.load(Ordering::Relaxed);
        (read(&part.undialed), read(&part.hashes_cut))
    };
    let cap = store.deciding(|world| world.spec().max_block_hashes());
    let mut id = 0;
    for decider in 0..2 {
        let model = (0..)
            .map(|n| format!("model-{n}"))
            .find(|model| bed.deciders.of(model) == decider)
            .expect("a model on each decider");
        ready(
            &store,
            &bed.reads.objects,
            &format!("m{decider}"),
            &model,
            Role::Single,
            WireModes::HTTP,
        );
        // One decision on the first decider, two on the second, and
        // the first of each with its hashes past the cap.
        for seed in 0..=decider {
            id += 1;
            let mut frame = constrained(frame_for(&model, id, seed as u64), 0b010, None);
            if seed == 0 {
                frame
                    .facts
                    .as_mut()
                    .expect("the case's frame has facts")
                    .block_hashes = (0..=cap as u64).collect();
            }
            let answer = bed.ask(frame).await;
            let shed = declined(proto::ShedReason::NoEndpoints, "wires");
            assert_eq!(outcome(&answer), &shed);
        }
    }
    assert_eq!(
        (part(&bed.counts, 0), part(&bed.counts, 1)),
        ((1, 1), (2, 1))
    );
    assert_eq!((bed.counts.undialed(), bed.counts.hashes_cut()), (3, 2));
    // A part stands on cache lines of its own, so two deciders' parts
    // share none.
    assert_eq!((size_of::<Part>(), align_of::<Part>()), (128, 128));
    let counts = DecideStats::default();
    for index in 0..PARTS + 2 {
        let part = counts.part(index);
        part.undialed.fetch_add(1 << index, Ordering::Relaxed);
        part.hashes_cut.fetch_add(3 << index, Ordering::Relaxed);
    }
    let whole = (1 << (PARTS + 2)) - 1;
    assert_eq!(part(&counts, 0), (1 + (1 << PARTS), 3 + (3 << PARTS)));
    assert_eq!(
        part(&counts, PARTS + 1),
        (2 + (2 << PARTS), 6 + (6 << PARTS))
    );
    assert_eq!((counts.undialed(), counts.hashes_cut()), (whole, 3 * whole));
}

/// The bed's frame under a session key.
fn keyed(key: u64, seed: u64) -> proto::DecideRequest {
    let mut frame = frame(u8::try_from(seed % 251).expect("under 251"), seed);
    frame
        .facts
        .as_mut()
        .expect("the case's frame has facts")
        .session_key = Some(proto::Identity {
        form: Some(proto::identity::Form::Hash(key)),
    });
    frame
}

/// Stands the bed's worker `worker` in `status`: on its object, as a
/// verdict leaves it, and the store is told.
fn set_status(bed: &Bed, worker: WorkerId, status: WorkerStatus) {
    stand(&bed.reads.objects, &bed.store, worker, status);
}

/// A worker told not ready, or draining, is passed over by the very
/// next decision, and the store publishes no world for its status: a
/// decision reads a worker's status on its worker object, so none waits
/// a publication for a worker that left rotation. Under `round_robin`
/// the three ready workers take requests by turns, so six in a row on
/// one worker show the two others passed over.
#[tokio::test]
async fn a_worker_told_out_of_rotation_is_passed_over_with_no_publication() {
    for out in [WorkerStatus::NotReady, WorkerStatus::Draining] {
        let mut bed = bed(Some(preset("round_robin", 1)), true, (1, 8), []);
        let stays = bed.workers[0];
        assert_eq!(bed.store.publish_due(), None, "no build waits");
        let published = bed.store.publications();
        for other in [bed.workers[1], bed.workers[2]] {
            set_status(&bed, other, out);
        }
        let told = bed.store.publications();
        assert_eq!(told, published, "{out:?}: a status published a world");
        for request in 1..=6 {
            let answer = bed.ask(frame(request, 1_000 + u64::from(request))).await;
            assert_eq!(picked(&answer), stays, "{out:?}: request {request}");
        }
        assert_eq!(bed.store.publications(), published, "{out:?}");
    }
}
