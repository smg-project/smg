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
    worker_host::{stand, Members, Objects},
};

impl DecisionSlots {
    /// What a decider's thread tells at its unwind, told by a case.
    pub(in crate::service::controller) fn tell_unwound(&self) {
        self.told.send_replace(true);
    }
}

pub(crate) mod encode;
mod named;

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
        let row = objects.record(id).expect("the record the write laid");
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

/// A snapshot as the load monitor publishes it for the bed's three
/// workers: the one at `idle` reports no queue, and the others are deep
/// in theirs.
fn reports(idle: usize) -> Arc<LoadSnapshot> {
    let loads = ["w0", "w1", "w2"].iter().enumerate();
    let loads = loads.map(|(at, name)| {
        let load = if at == idle {
            one_rank_load(0, 0.1, 10.0)
        } else {
            one_rank_load(900_000, 0.9, 10.0)
        };
        (format!("http://{name}:8000"), load)
    });
    LoadSnapshot::from_loads_for_test(loads.collect())
}

/// A deciding thread gives each frame's request the newest snapshot
/// the load monitor published. Under the least-load preset every pick
/// names the one worker whose report shows no queue. The monitor then
/// publishes another worker so, and the picks follow at the next frame.
/// It publishes an empty snapshot, as at its lead's end, and the next
/// frame reads no report of the one before: its pick is the one worker
/// that holds no request.
#[tokio::test]
async fn a_decision_reads_the_reports_the_load_monitor_last_published() {
    let mut bed = bed(Some(preset("least_load", 1)), true, (2, 8), []);
    // Three frames a publication: a worker of the bed runs four requests
    // at the most.
    for (idle, ids) in [(0, 1..=3u8), (1, 4..=6)] {
        bed.loads.send_replace(reports(idle));
        for id in ids {
            let answer = bed.ask(frame(id, 100 + u64::from(id))).await;
            assert_eq!(picked(&answer), bed.workers[idle], "{answer:?}");
        }
    }
    bed.loads.send_replace(Arc::new(LoadSnapshot::default()));
    let answer = bed.ask(frame(7, 107)).await;
    assert_eq!(picked(&answer), bed.workers[2], "{answer:?}");
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

/// A snapshot as the load monitor publishes it for the bed's three
/// workers, each report counting its resident tokens: the one at `idle`
/// holds none, and the others hold many.
fn resident_reports(idle: usize) -> Arc<LoadSnapshot> {
    let loads = ["w0", "w1", "w2"].iter().enumerate();
    let loads = loads.map(|(at, name)| {
        let mut load = one_rank_load(0, 0.1, 10.0);
        load.loads[0].num_resident_tokens = Some(if at == idle { 0 } else { 900_000 });
        (format!("http://{name}:8000"), load)
    });
    LoadSnapshot::from_loads_for_test(loads.collect())
}

/// A document whose scorer is `resident-work` is given the snapshot, as
/// any document that reads reports: the scorer keeps no number of a
/// report. The monitor publishes one worker that holds no resident
/// tokens and the others many, and three frames name that worker,
/// whatever it holds in flight by then. The monitor then publishes
/// another worker so, and the picks follow. Given no snapshot, the
/// scorer would cost each worker by its requests in flight, and three
/// frames would name three workers.
#[tokio::test]
async fn a_decision_under_the_resident_work_scorer_reads_the_monitors_reports() {
    let document: PolicySpec = serde_json::from_str(
        r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": {
                "score": [{ "resident-work": { "request_tokens": 512 } }],
                "pick": "best"
              } },
              "bindings": [{ "profile": "p" }]
            }"#,
    )
    .expect("a document of this build");
    let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
    // Three frames a publication: a worker of the bed runs four requests
    // at the most.
    for (idle, ids) in [(0, 1..=3u8), (1, 4..=6)] {
        bed.loads.send_replace(resident_reports(idle));
        for id in ids {
            let answer = bed.ask(frame(id, 100 + u64::from(id))).await;
            assert_eq!(picked(&answer), bed.workers[idle], "{answer:?}");
        }
    }
}

/// A document with a stage that keeps load reports is given no
/// snapshot: its engine's own task tells such a stage each report, and
/// a controller runs no such task yet. The stage is the cases' own
/// ([`KeepsReports`]). The monitor publishes one worker with no queue
/// and the others deep in theirs, and three frames still name three
/// workers, each by the requests it holds. The scorer that reads
/// reports outweighs the one that keeps them eight to one: given the
/// snapshot, it would name the worker with no queue three times.
#[tokio::test]
async fn a_document_with_a_stage_that_keeps_reports_is_given_no_snapshot() {
    let document: PolicySpec = serde_json::from_str(
        r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": {
                "score": [
                  { "least-load": { "weight": 8 } },
                  "keeps-reports"
                ],
                "pick": "best"
              } },
              "bindings": [{ "profile": "p" }]
            }"#,
    )
    .expect("a document of this build");
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer("keeps-reports", || Box::new(KeepsReports))
        .expect("a free name");
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    bed.loads.send_replace(reports(0));
    let mut named = Vec::new();
    for id in 1..=3u8 {
        let answer = bed.ask(frame(id, 100 + u64::from(id))).await;
        named.push(picked(&answer));
    }
    assert!(
        named[0] != named[1] && named[1] != named[2] && named[0] != named[2],
        "the picks followed a report: {named:?}"
    );
}

/// The service tells no stage the issue of a request its sessions rule
/// keeps from the profile. A key's first request is placed by the
/// profile's picker, which hears its issue. The key's next request
/// follows its pin: under `untold` its lease is filed as kept, and the
/// picker hears nothing of it ahead of its release. Under `told`, the
/// default, the picker hears that issue as any other.
#[tokio::test]
async fn the_service_tells_no_stage_the_issue_of_a_request_its_rule_keeps() {
    use crate::{
        document::{Assignment, RulePlaced},
        service::controller::lifecycle::tests::{a_dispatched, a_released},
    };
    for kept in [RulePlaced::Untold, RulePlaced::Told] {
        let told = Arc::new(Mutex::new(Calls::new()));
        let hears = Arc::clone(&told);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("told", move || Box::new(Told(Arc::clone(&hears))))
            .expect("a free name");
        // The manual preset's rule over a profile that picks through
        // the case's stage: a new key is the profile's to place.
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = 1;
        let profile = document.profiles.get_mut("manual");
        profile.expect("the preset's profile").pick = "told".into();
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        (rule.assignment, rule.rule_placed) = (Assignment::Delegate, kept);
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the catalog holds the stage");
        let mut bed = bed(Some(spec), true, (1, 8), []);
        let at = bed.script.taken;
        let untold = |bed: &Bed, answer: &proto::DecideAnswer| {
            let id = LeaseId(lease_of(answer).id);
            let filed = LeasesStore::get(&*bed.store, GATEWAY, id);
            filed.expect("the lease is on file").untold
        };
        let first = bed.ask(keyed(1, 1_001)).await;
        assert!(!untold(&bed, &first), "{kept:?}: a new key");
        let hand = Hand::new(GATEWAY);
        lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&first).id), at)
            .expect("the frame reads");
        let pinned = bed.ask(keyed(1, 1_002)).await;
        assert_eq!(picked(&pinned), picked(&first), "{kept:?}: its pin");
        assert_eq!(untold(&bed, &pinned), kept == RulePlaced::Untold);
        let (first, pinned) = (lease_of(&first).id, lease_of(&pinned).id);
        lifecycle::released(&bed.store, &hand, &a_released(pinned)).expect("the frame reads");
        let mut heard = vec![(first, None)];
        if kept == RulePlaced::Told {
            heard.push((pinned, None));
        }
        heard.push((pinned, Some(ReleaseReason::ClientGone)));
        // The hand leaves the release to a decider still at work
        // behind its answer, which tells it as it leaves that work.
        settled(|| told.lock().len() >= heard.len()).await;
        assert_eq!(*told.lock(), heard, "{kept:?}");
    }
}

/// A pinned request is none of the requests its policy sent, through
/// the service as in a gateway's own process: the case of the same name
/// in `tests/classic/sticky_override.rs`, asked by frames, with its
/// counts. Two workers each hold one request, a lease on file, and the
/// second reports 5,000 tokens waiting. A key's first request of 1,000
/// tokens goes to the first worker, placed by the policy, and its next
/// request of 10,000 tokens follows the pin: its lease is filed as kept
/// from the profile. The policy then prices the first worker at the
/// 1,000 tokens it placed there and at two requests more on file, at
/// the mean prompt of 1,024 tokens: 3,048, under the second worker's
/// 6,024. So a request with no key goes to the first worker. Counted as
/// a send of the policy's, the pinned request would price that worker
/// at 12,024.
///
/// The pinned request still weighs what a lease on file weighs: with
/// 1,500 tokens waiting on the second worker, 2,524 in all, the request
/// with no key goes there. Once the pinned request ends it weighs
/// nothing: the first worker stands at 2,024, and the request with no
/// key goes back to it.
#[tokio::test]
async fn a_served_pinned_request_is_none_of_the_requests_its_policy_sent() {
    use crate::{
        document::{
            Assignment, LoadMetricSetting, PolicySettings, RoutingKeyOverrideSettings,
            RoutingSettings,
        },
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::tests::{a_completed, a_dispatched, a_released},
        store::{LeaseRecord, PickRecord},
    };
    let policies = [
        PolicySettings::LeastLoad {
            kv_pressure_weight: 0.15,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
        },
        PolicySettings::PowerOfTwo {
            load_metric: LoadMetricSetting::ExpectedWait,
        },
    ];
    for policy in policies {
        let name = policy.name();
        // The settings' document for the policy under the sticky
        // override: its rule keeps what it places from the profile.
        let settings = RoutingSettings {
            policy,
            routing_key_override: RoutingKeyOverrideSettings {
                enabled: true,
                headers: vec!["x-smg-routing-key".into()],
                eviction_interval_secs: 60,
                max_idle_secs: 14_400,
                assignment_mode: Assignment::Delegate,
            },
            ..RoutingSettings::default()
        };
        let mut document = settings.policy_document().expect("the settings' document");
        document.generation = 1;
        let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
        let at = bed.script.taken;
        let (near, far, aside) = (bed.workers[0], bed.workers[1], bed.workers[2]);
        // Two workers take requests: the third is set not ready.
        stand(
            &bed.reads.objects,
            &bed.store,
            aside,
            WorkerStatus::NotReady,
        );
        // One request on each of the two, held by a lease on file that
        // no decision of the policy's filed.
        for (id, target) in [(1, near), (2, far)] {
            let held = LeaseRecord {
                lease: Lease {
                    id: LeaseId(1 << 63 | id),
                    ..Lease::default()
                },
                gateway: GATEWAY,
                request: id,
                target,
                picked: PickRecord::default(),
                decode: None,
                untold: false,
                term: 1,
                issued_at: at,
                dispatch_deadline: at + DISPATCH_TTL,
                state: LeaseState::ISSUED,
            };
            assert_eq!(LeasesStore::issue(&*bed.store, held), Ok(()));
        }
        // The two workers' reports, the second with `waiting` tokens in
        // its queue, sampled ahead of every send: none reflects one.
        let reports = |waiting: i32| {
            let loads = [("w0", 0), ("w1", waiting)].map(|(worker, waiting)| {
                let mut report = one_rank_load(waiting, 0.1, 1_000.0);
                report.sampled_at = Some(at - Duration::from_millis(1));
                (format!("http://{worker}:8000"), report)
            });
            LoadSnapshot::from_loads_for_test(loads.into_iter().collect())
        };
        let sized = |mut frame: proto::DecideRequest, tokens: u32| {
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.prompt_tokens = tokens;
            frame
        };
        let filed = |bed: &Bed, answer: &proto::DecideAnswer| {
            let id = LeaseId(lease_of(answer).id);
            let filed = LeasesStore::get(&*bed.store, GATEWAY, id);
            let filed = filed.expect("the lease is on file");
            (filed.lease.session_verdict, filed.untold)
        };
        let hand = Hand::new(GATEWAY);
        let sent = |bed: &Bed, answer: &proto::DecideAnswer| {
            let frame = a_dispatched(lease_of(answer).id);
            lifecycle::dispatched(&bed.store, &hand, &frame, at).expect("the frame reads");
        };
        let given_up = |bed: &Bed, answer: &proto::DecideAnswer| {
            let frame = a_released(lease_of(answer).id);
            lifecycle::released(&bed.store, &hand, &frame).expect("the frame reads");
        };
        bed.loads.send_replace(reports(5_000));
        let first = bed.ask(sized(keyed(1, 1_001), 1_000)).await;
        assert_eq!(picked(&first), near, "{name}: a new key");
        assert_eq!(filed(&bed, &first), (SessionVerdict::Claimed(near), false));
        sent(&bed, &first);
        let pinned = bed.ask(sized(keyed(1, 1_002), 10_000)).await;
        assert_eq!(picked(&pinned), near, "{name}: its pin");
        assert_eq!(filed(&bed, &pinned), (SessionVerdict::Pinned, true));
        sent(&bed, &pinned);
        let plain = bed.ask(sized(frame(3, 1_003), 1_000)).await;
        assert_eq!(
            picked(&plain),
            near,
            "{name}: the pinned request is no send of the policy's"
        );
        given_up(&bed, &plain);
        bed.loads.send_replace(reports(1_500));
        let plain = bed.ask(sized(frame(4, 1_004), 1_000)).await;
        assert_eq!(
            picked(&plain),
            far,
            "{name}: the pinned request weighs as one on file"
        );
        given_up(&bed, &plain);
        let ended = a_completed(lease_of(&pinned).id, near);
        lifecycle::completed(&bed.store, &hand, &ended).expect("the frame reads");
        let plain = bed.ask(sized(frame(5, 1_005), 1_000)).await;
        assert_eq!(
            picked(&plain),
            near,
            "{name}: an ended pinned request weighs nothing"
        );
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

/// The upkeep of a served engine rides the presence sweep's task: that
/// engine runs no task of its own. A session key a decision read leaves
/// the engine's table by the task's own tick once its idle time ended,
/// and no sooner, with no request behind the one that read it.
#[tokio::test]
async fn the_sweeps_task_takes_a_session_key_out_once_its_idle_time_ended() {
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = 1;
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    (rule.ttl_s, rule.sweep_s) = (1, 1);
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let keys = |bed: &Bed| {
        bed.store
            .deciding(|world| world.spec().engine().sessions().len())
    };
    bed.ask(keyed(1, 1_001)).await;
    assert_eq!(keys(&bed), 1, "the key its decision read");

    let grace = Duration::from_millis(60);
    let fence = bed.elector.fence();
    let gateways = Arc::new(Gateways::new(grace, 1_024, fence, Arc::clone(&bed.store)));
    let (stop, stopped) = watch::channel(false);
    let mut task = JoinSet::new();
    task.spawn(async move { gateways.run(stopped).await });
    // The decision is dated at the bed's take.
    let read = bed.script.taken;
    while keys(&bed) > 0 {
        assert!(read.elapsed() < PATIENCE, "the task took no key out");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(
        read.elapsed() >= Duration::from_secs(1),
        "the key left inside its idle time"
    );
    stop.send_replace(true);
    timeout(PATIENCE, task.join_next())
        .await
        .expect("the task ends at its stop")
        .expect("one task")
        .expect("it ended by itself");
}

/// A served engine's session table hears the store's word of a worker
/// that leaves and of its return. The worker a key is pinned on is
/// removed through the store: the table lists it absent. Registered
/// again under its id, it is listed no more.
#[tokio::test]
async fn the_session_table_lists_a_worker_the_store_removed_until_it_returns() {
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = 1;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let absent = |bed: &Bed| {
        bed.store
            .deciding(|world| world.spec().engine().sessions().absent())
    };
    let pinned = picked(&bed.ask(keyed(1, 1_001)).await);
    assert_eq!(absent(&bed), 0, "every worker stands");
    let row = WorkersStore::get(&*bed.store, pinned).expect("the row");
    let objects = &bed.reads.objects;
    let removed = objects.remove(&bed.store, pinned, row.revision);
    assert!(removed.is_some(), "the store took the worker out");
    assert_eq!(absent(&bed), 1, "the worker its key is pinned on left");
    assert_eq!(rewritten(&bed, row.facts), pinned, "its id again");
    assert_eq!(absent(&bed), 0, "back under its id");
}

/// A decision waits in its stream's queue as it was made, with the
/// lease issued for its pick and what is left of that lease's span: its
/// frame is its stream's to build. A `Busy` is handed over as its
/// frame.
#[tokio::test]
async fn a_decision_is_handed_to_its_stream_as_it_was_made_and_a_busy_as_its_frame() {
    let steps = [Duration::ZERO, Duration::ZERO, BUDGET].map(Step::At);
    let mut bed = leading(steps);
    assert_eq!(bed.submit(frame(1, 1)), None);
    let handed = timeout(PATIENCE, bed.answers.recv()).await;
    let Ok(Some(wire::Decided::Decision {
        id,
        term,
        decision,
        lease: Some(leased),
        encode: None,
    })) = handed
    else {
        panic!("a decision as it was made, with its lease: {handed:?}");
    };
    assert_eq!((id.as_slice(), term), (&[1][..], 1));
    let first = decision.pick().and_then(|pick| pick.targets.first());
    let first = first.expect("a pick names a target");
    assert!(bed.workers.contains(&first.id), "{decision:?}");
    assert_eq!(leased.lease.id, LeaseId(1 << 32 | 1));
    assert_eq!(leased.left, DISPATCH_TTL);
    // A request taken with its budget gone is answered where it is
    // taken, and the answer carries no list.
    assert_eq!(bed.submit(frame(2, 2)), None);
    let handed = timeout(PATIENCE, bed.answers.recv()).await;
    let Ok(Some(wire::Decided::Frame(told))) = handed else {
        panic!("a Busy as its frame: {handed:?}");
    };
    assert_eq!(busy(&told), Reason::BudgetGoneOnArrival);
    assert_eq!(told.request_id, [2]);
}

/// A member's requests in flight are the leases on file that book it:
/// a decision by the fewest requests goes where the decisions before
/// it left the fewest, and a lease that ends gives its worker the next.
#[tokio::test]
async fn a_decision_by_the_fewest_requests_reads_the_leases_the_decisions_before_it_left() {
    let mut document = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
        .expect("a form of this build's preset");
    document.generation = 1;
    let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
    // Nine decisions and no word from their gateway: each round of
    // three names each of the three members once.
    let mut held = Vec::new();
    for round in 0..3u8 {
        let mut named = BTreeSet::new();
        for turn in 1..=3u8 {
            let id = round * 3 + turn;
            let answer = bed.ask(frame(id, u64::from(id))).await;
            named.insert(picked(&answer));
            held.push((picked(&answer), LeaseId(lease_of(&answer).id)));
        }
        assert_eq!(named.len(), 3, "round {round} named {named:?}");
    }
    // One member's three leases end at their gateway's word: the next
    // three decisions are that member's.
    let (freed, _) = held[0];
    for &(_, lease) in held.iter().filter(|(worker, _)| *worker == freed) {
        let ended = bed
            .store
            .released(GATEWAY, lease, None, ReleaseReason::ClientGone);
        assert!(ended, "{lease:?} was on file");
    }
    for id in 10..=12u8 {
        let answer = bed.ask(frame(id, u64::from(id))).await;
        assert_eq!(picked(&answer), freed, "request {id}");
    }
}

/// A decision reads its rows behind what was handed while its decider
/// worked: a release handed while the decider is held at a request is
/// applied by that request ahead of its reads. Under the fewest
/// requests the request then picks as the first one did under its
/// seed: that worker stands at no request again.
#[tokio::test]
async fn a_decision_reads_its_rows_behind_what_was_handed_ahead_of_it() {
    let mut document = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
        .expect("a form of this build's preset");
    document.generation = 1;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let first = bed.ask(frame(1, 7)).await;
    // The second request is held at its read of the clock: its decider
    // is at work, and what is handed is left to it.
    let (reached, held) = std_mpsc::sync_channel(1);
    let (go, word) = std_mpsc::sync_channel(1);
    let hold = Step::Hold { reached, go: word };
    bed.script.steps.lock().push_back(hold);
    assert_eq!(bed.submit(frame(2, 7)), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    let hand = Hand::new(GATEWAY);
    bed.store.hand(&hand, LeaseId(lease_of(&first).id), gone);
    let live = LeasesStore::live(&*bed.store, GATEWAY);
    assert_eq!(live, 1, "the hand leaves the release to the decider");
    go.send(()).expect("the decider holds the other end");
    let second = bed.answer().await;
    assert_eq!(picked(&second), picked(&first), "no request stands there");
}

#[tokio::test]
async fn a_picks_answer_carries_the_lease_the_store_filed_under_the_fences_term() {
    let mut bed = leading([]);
    let answer = bed.ask(frame(7, 7)).await;
    let lease = *lease_of(&answer);
    assert_eq!(lease.id, 1 << 32 | 1, "the term over its first count");
    assert_eq!(
        (lease.term, lease.origin(), lease.kind(), lease.side()),
        (
            answer.term,
            proto::LeaseOrigin::Controller,
            proto::LeaseKind::Credit,
            proto::Side::Single
        )
    );
    assert_eq!(
        (lease.target_index, lease.generation, lease.session_key),
        (0, 1, 0)
    );
    assert_eq!(u128::from(lease.dispatch_ttl_ms), DISPATCH_TTL.as_millis());
    let store: &dyn LeasesStore = bed.store.as_ref();
    let filed = store
        .get(GATEWAY, LeaseId(lease.id))
        .expect("the lease is on file");
    assert_eq!(
        (filed.target, filed.term, filed.state, filed.request),
        (picked(&answer), 1, LeaseState::ISSUED, fingerprint(&[7]))
    );
    assert_eq!(filed.dispatch_deadline, filed.issued_at + DISPATCH_TTL);
    // The next decision's id is the term's next count, its lease
    // carries its request's prompt tokens, and both leases stand for
    // their gateway alone.
    assert_eq!(lease.tokens, 0, "a request that states no prompt");
    let mut prompted = frame(8, 8);
    let facts = prompted.facts.as_mut();
    let facts = facts.expect("the case's frame has facts");
    facts.prompt_tokens = 48;
    facts.session_key = Some(proto::Identity {
        form: Some(proto::identity::Form::Hash(17)),
    });
    let next = bed.ask(prompted).await;
    assert_eq!(
        (lease_of(&next).id, lease_of(&next).tokens),
        (1 << 32 | 2, 48)
    );
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!((store.live(GATEWAY), store.live(OTHER)), (2, 0));
    // The lease carries its request's session, on the wire as on file:
    // never zero for a request that states one.
    let session = lease_of(&next).session_key;
    let filed = store.get(GATEWAY, LeaseId(1 << 32 | 2));
    let filed = filed.expect("the lease is on file").lease;
    assert_ne!(session, 0);
    assert_eq!(filed.session.get(), session);
}

/// A frame's client key places its request on the ring: one client's
/// requests take one worker whatever their seeds, and clients spread
/// over the workers.
#[tokio::test]
async fn a_frames_client_key_places_its_request_on_the_ring() {
    let mut bed = bed(Some(preset("consistent_hashing", 1)), true, (2, 8), []);
    let from = |id: u8, client: u64| {
        let mut frame = frame(id, u64::from(id));
        frame
            .facts
            .as_mut()
            .expect("the frame has facts")
            .client_key = client;
        frame
    };
    let mut of_one = Vec::new();
    for id in 1..=12u8 {
        of_one.push(picked(&bed.ask(from(id, 0x9e37_79b9_7f4a_7c15)).await));
    }
    assert!(
        of_one.windows(2).all(|pair| pair[0] == pair[1]),
        "{of_one:?}"
    );
    let mut of_many = Vec::new();
    for id in 13..=28u8 {
        let client = u64::from(id).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        of_many.push(picked(&bed.ask(from(id, client)).await));
    }
    assert!(
        of_many.windows(2).any(|pair| pair[0] != pair[1]),
        "{of_many:?}"
    );
}

/// The worker `frame` takes on `bed`. Its lease is given up where it
/// stands, so the request weighs on no later one.
async fn placed(bed: &mut Bed, frame: proto::DecideRequest) -> WorkerId {
    use crate::service::controller::lifecycle::tests::a_released;
    let answer = bed.ask(frame).await;
    let released = a_released(lease_of(&answer).id);
    lifecycle::released(&bed.store, &Hand::new(GATEWAY), &released).expect("the frame reads");
    picked(&answer)
}

/// A frame says who handed its session key, and the prefix form of a
/// ring reads the word. The document is the settings' for `prefix_hash`
/// under the sticky override. Twelve keys under one prompt, each
/// handed by its request's own id, take the worker a request of that
/// prompt takes with no key: the ring places them by the prompt. Twelve
/// keys under the same prompt whose frames say nothing lead it, and
/// stand on more workers than that one: a header's keys, and every key
/// of a gateway older than the field. The id's key names the session
/// still: once dispatched, its next request follows its pin under a
/// prompt another worker owns.
#[tokio::test]
async fn the_prefix_ring_places_a_frame_by_its_prompt_where_its_id_handed_the_key() {
    use crate::{
        document::{PolicySettings, RoutingKeyOverrideSettings, RoutingSettings},
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::tests::a_dispatched,
    };
    let settings = RoutingSettings {
        policy: PolicySettings::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: Vec::new(),
        },
        routing_key_override: RoutingKeyOverrideSettings {
            enabled: true,
            ..RoutingKeyOverrideSettings::default()
        },
        ..RoutingSettings::default()
    };
    let mut document = settings.policy_document().expect("the settings' document");
    document.generation = 1;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    // A frame of four blocks from `prompt` on, sixteen tokens a block,
    // under `key` where it has one, and `by_id` is the word it says of
    // that key.
    let asked = |key: Option<u64>, by_id: bool, prompt: u64, seed: u64| {
        let mut one = match key {
            Some(key) => keyed(key.wrapping_mul(0x9e37_79b9_7f4a_7c15), seed),
            None => frame(u8::try_from(seed % 251).expect("under 251"), seed),
        };
        let facts = one.facts.as_mut().expect("the case's frame has facts");
        facts.block_hashes = (prompt..prompt + 4).collect();
        facts.prompt_tokens = 64;
        facts.session_key_from_rid = by_id;
        one
    };
    // Two prompts that two workers own: a request with no key goes by
    // its prompt.
    const NEAR: u64 = 1_000;
    let here = placed(&mut bed, asked(None, false, NEAR, 1)).await;
    let (mut far, mut there) = (NEAR, here);
    while there == here {
        far += 100;
        assert!(far < 4_000, "every prompt stands on one worker");
        there = placed(&mut bed, asked(None, false, far, far)).await;
    }
    // The requests' own ids handed these keys: each goes where its
    // prompt goes.
    for key in 1..=12 {
        let by_id = placed(&mut bed, asked(Some(key), true, NEAR, 100 + key)).await;
        assert_eq!(by_id, here, "key {key}");
    }
    // These frames say nothing of their keys, and each key leads the
    // prompt.
    let mut led = Vec::new();
    for key in 13..=24 {
        led.push(placed(&mut bed, asked(Some(key), false, NEAR, 100 + key)).await);
    }
    assert!(led.iter().any(|&worker| worker != here), "{led:?}");
    // An id's key is its session's name: dispatched, it pins its
    // worker, and its next request follows the pin under the prompt
    // the other worker owns.
    let first = bed.ask(asked(Some(25), true, NEAR, 125)).await;
    assert_eq!(picked(&first), here);
    let (hand, at) = (Hand::new(GATEWAY), bed.script.taken);
    lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&first).id), at)
        .expect("the frame reads");
    let next = bed.ask(asked(Some(25), true, far, 126)).await;
    assert_eq!(picked(&next), here, "its pin, and not {there:?}");
    let filed = LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(&next).id));
    let filed = filed.expect("the lease is on file").lease;
    assert_eq!(filed.session_verdict, SessionVerdict::Pinned);
}

/// The plain ring reads no word of who handed a key. Under
/// `consistent_hashing` a frame that says its request's own id handed
/// its key takes the worker that key takes with nothing said, and the
/// keys stand on more than one worker.
#[tokio::test]
async fn the_plain_ring_places_a_frame_by_its_key_whoever_handed_it() {
    let mut bed = bed(Some(preset("consistent_hashing", 1)), true, (2, 8), []);
    let said = |key: u64, by_id: bool| {
        let mut one = keyed(key.wrapping_mul(0x9e37_79b9_7f4a_7c15), key);
        let facts = one.facts.as_mut().expect("the case's frame has facts");
        facts.session_key_from_rid = by_id;
        one
    };
    let mut by_key = Vec::new();
    for key in 1..=12 {
        let silent = placed(&mut bed, said(key, false)).await;
        let by_id = placed(&mut bed, said(key, true)).await;
        assert_eq!(by_id, silent, "key {key}");
        by_key.push(silent);
    }
    assert!(
        by_key.windows(2).any(|pair| pair[0] != pair[1]),
        "{by_key:?}"
    );
}

/// Requests that state no prompt size spread over the members: each
/// dispatch is its worker's credit at the load stage's mean until that
/// worker's next report, and each lease states no size.
#[tokio::test]
async fn a_dispatched_request_that_stated_no_size_is_priced_until_its_workers_next_report() {
    let mut bed = bed(Some(preset("least_load", 1)), true, (2, 8), []);
    // Each member's report, sampled ahead of every send, at one rate:
    // the first holds no work and the other two a quarter of a mean
    // prompt each. The load monitor publishes the three.
    let at = bed.script.taken;
    let loads = ["w0", "w1", "w2"].iter().enumerate();
    let loads = loads.map(|(turn, name)| {
        let mut report = one_rank_load(if turn == 0 { 0 } else { 256 }, 0.1, 1_000.0);
        report.sampled_at = Some(at - Duration::from_millis(1));
        (format!("http://{name}:8000"), report)
    });
    bed.loads
        .send_replace(LoadSnapshot::from_loads_for_test(loads.collect()));
    // Three requests that state no prompt size, each dispatched before
    // the next is decided. A dispatch is its worker's credit until that
    // worker's next report, and the load stage credits one of no stated
    // size at its mean: at no tokens it would cost nothing there, and
    // each request would read the first member as free.
    let mut named = Vec::new();
    for id in 1..=3u8 {
        let answer = bed.ask(frame(id, u64::from(id))).await;
        assert_eq!(
            lease_of(&answer).tokens,
            0,
            "the lease states what the request did"
        );
        named.push(picked(&answer));
        let frame = proto::Dispatched {
            lease_id: lease_of(&answer).id,
            target_index: 0,
            side: proto::Side::Single.into(),
            since_decision_micros: 250,
        };
        let hand = Hand::new(GATEWAY);
        lifecycle::dispatched(&bed.store, &hand, &frame, bed.script.taken)
            .expect("the frame reads");
    }
    assert_eq!(named[0], bed.workers[0], "the member with no work");
    let members: BTreeSet<WorkerId> = named.iter().copied().collect();
    assert_eq!(members.len(), 3, "each member took one request: {named:?}");
}

#[tokio::test]
async fn an_answer_that_names_no_worker_or_does_not_leave_files_no_lease() {
    // A shed: no member serves the model.
    let mut bed = leading([]);
    let answer = bed.ask(frame_for("model-none", 1, 1)).await;
    assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
    assert_eq!(unleased(answer.clone()), answer, "a shed carries no lease");
    assert_eq!(bed.store.live(GATEWAY), 0);
    // A decision whose fence closed at its hand-over, and one whose
    // budget ended while it was made: `Busy`, and nothing on file.
    for (second_read, reason) in [
        (FENCED, Reason::NotLeader),
        (BUDGET, Reason::BudgetGoneOnArrival),
    ] {
        let mut bed = leading([Step::At(Duration::ZERO), Step::At(second_read)]);
        assert_eq!(busy(&bed.ask(frame(2, 2)).await), reason);
        assert_eq!(bed.store.live(GATEWAY), 0, "{reason:?}");
    }
}

#[tokio::test]
async fn a_pick_the_store_mints_no_lease_for_leaves_as_busy() {
    // The term's last id, then none: `Busy` with no reason, counted.
    let mut bed = leading([]);
    bed.store.spend_ids(1);
    let last = bed.ask(frame(1, 1)).await;
    assert_eq!(lease_of(&last).id, 1 << 32 | u64::from(u32::MAX));
    for id in 2..=3 {
        assert_eq!(busy(&bed.ask(frame(id, 1)).await), Reason::Unspecified);
    }
    assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 2);
    assert_eq!(bed.store.live(GATEWAY), 1, "the last id's lease alone");
    // A store that stands under a later term than the fence showed:
    // not this leader's to answer, and no lease.
    let mut bed = leading([]);
    bed.store.set_term(2);
    assert_eq!(busy(&bed.ask(frame(4, 1)).await), Reason::NotLeader);
    assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 0);
    assert_eq!(bed.store.live(GATEWAY), 0);
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

/// The answer that carries the id at a term's mark says the word to
/// leave, only once it is handed over, and no id past it says another.
#[tokio::test]
async fn the_id_at_a_terms_mark_says_the_word_once_its_answer_is_handed_over() {
    let mut bed = leading([]);
    let (words, go) = heard(&bed);
    // The mark is three quarters of a term's ids. The id under it, the
    // mark's, and three past it.
    assert_eq!(u64::from(TERM_MARK) * 4, 3 << 32);
    bed.store.spend_ids(u32::MAX - TERM_MARK + 2);
    let under = bed.ask(frame(1, 1)).await;
    assert_eq!(lease_of(&under).id, 1 << 32 | u64::from(TERM_MARK - 1));
    // The decider is held inside the word: an answer that reaches the
    // case was handed over before the word was said.
    let at = bed.ask(frame(2, 1)).await;
    assert_eq!(lease_of(&at).id, 1 << 32 | u64::from(TERM_MARK));
    words
        .recv_timeout(PATIENCE)
        .expect("the mark's decider says the word");
    // A later word finds its go waiting. The ids past the mark say
    // none: the answer after each shows its decider said nothing.
    for _ in 0..8 {
        go.send(()).expect("the decider holds the other end");
    }
    for (id, past) in [(3, 1), (4, 2), (5, 3)] {
        let answer = bed.ask(frame(id, 1)).await;
        assert_eq!(lease_of(&answer).id, 1 << 32 | u64::from(TERM_MARK + past));
    }
    assert!(words.try_recv().is_err(), "one answer a term says the word");
    assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 0);
}

/// The mark's own id need not be one an answer carries: the mint steps
/// over an id on file. The first answer past the mark says the word
/// then, and once.
#[tokio::test]
async fn an_id_past_a_terms_mark_says_the_word_the_marks_own_never_said() {
    let mut bed = leading([]);
    let (words, go) = heard(&bed);
    for _ in 0..8 {
        go.send(()).expect("the decider holds the other end");
    }
    // The term's counter stands two ids past its mark.
    bed.store.spend_ids(u32::MAX - TERM_MARK - 2);
    let first = bed.ask(frame(1, 1)).await;
    assert_eq!(lease_of(&first).id, 1 << 32 | u64::from(TERM_MARK + 3));
    words
        .recv_timeout(PATIENCE)
        .expect("the first answer past the mark says the word");
    // The answer after each shows its decider said nothing.
    for (id, past) in [(2, 4), (3, 5)] {
        let answer = bed.ask(frame(id, 1)).await;
        assert_eq!(lease_of(&answer).id, 1 << 32 | u64::from(TERM_MARK + past));
    }
    assert!(words.try_recv().is_err(), "one answer a term says the word");
}

/// Each term says its own word: a leader past its mark under one term,
/// then under the next, hands the lease over twice.
#[tokio::test]
async fn two_terms_in_a_row_each_say_the_word_at_their_mark() {
    let mut bed = leading([]);
    let (words, go) = heard(&bed);
    let taken = bed.script.taken;
    for term in 1..=2 {
        go.send(()).expect("the decider holds the other end");
        if term > 1 {
            // The bed's candidate, alone over its lease, leaves and
            // takes again at one instant, as its election would.
            let left = bed.elector.step_down(None, taken);
            assert!(left.expect("a planned hand-over"));
            let took = bed.elector.tick(taken).expect("the take");
            assert_eq!(took, TickOutcome::Acquired { term });
            bed.store.set_term(term);
        }
        bed.store.spend_ids(u32::MAX - TERM_MARK + 1);
        let at = bed.ask(frame(1, 1)).await;
        assert_eq!(lease_of(&at).id, term << 32 | u64::from(TERM_MARK));
        // The answer after the mark's shows its decider said its word.
        let past = bed.ask(frame(2, 1)).await;
        assert_eq!(lease_of(&past).id, term << 32 | u64::from(TERM_MARK + 1));
        assert!(words.try_recv().is_ok(), "the word under term {term}");
        assert!(words.try_recv().is_err(), "one word under term {term}");
    }
}

/// Under the last term a replica can take, the mark says no word: a
/// hand-over there leaves no term to take. The term before it says its
/// own.
#[tokio::test]
async fn the_mark_of_the_last_term_says_no_word() {
    let mut bed = leading([]);
    let (words, go) = heard(&bed);
    go.send(()).expect("the decider holds the other end");
    // The bed's candidate, alone over its lease, leaves and takes again
    // at one instant, twice: above a term said, the term before the
    // last; then at a planned hand-over, the last under the ceiling.
    let (last, taken) = (TERM_CEILING - 1, bed.script.taken);
    for (term, above, says) in [(last - 1, Some(last - 2), true), (last, None, false)] {
        let left = bed.elector.step_down(above, taken);
        assert!(left.expect("a term that leaves one to take"));
        let took = bed.elector.tick(taken).expect("the take");
        assert_eq!(took, TickOutcome::Acquired { term });
        bed.store.set_term(term);
        bed.store.spend_ids(u32::MAX - TERM_MARK + 1);
        let at = bed.ask(frame(1, 1)).await;
        assert_eq!(lease_of(&at).id, term << 32 | u64::from(TERM_MARK));
        // The answer after the mark's shows its decider said what it
        // had to say.
        let past = bed.ask(frame(2, 1)).await;
        assert_eq!(lease_of(&past).id, term << 32 | u64::from(TERM_MARK + 1));
        assert_eq!(words.try_recv().is_ok(), says, "under term {term}");
    }
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

/// [`Told`], whose issue call waits for its case: what falls while a
/// stage's own call runs is the case's to say.
#[derive(Debug)]
struct Waits {
    told: Told,
    reached: std_mpsc::SyncSender<()>,
    go: Arc<Mutex<std_mpsc::Receiver<()>>>,
    /// Its release call panics.
    bursts: bool,
}

impl Lifecycle for Waits {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        self.told.on_decision_issued(lease, decision);
        self.reached.send(()).expect("the case waits for this call");
        let word = self.go.lock().recv_timeout(PATIENCE);
        word.expect("the case lets the call go on");
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        assert!(!self.bursts, "the case's stage panics at a release");
        self.told.on_released(lease, reason);
    }
}

impl Picker for Waits {
    fn name(&self) -> &'static str {
        "waits"
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

/// A leader whose one picker waits inside its issue call: the bed, what
/// the engine told the stage of leases, and the case's two ends of the
/// wait, the call reached and the word that lets it go on.
fn waiting() -> (
    Bed,
    Arc<Mutex<Calls>>,
    std_mpsc::Receiver<()>,
    std_mpsc::SyncSender<()>,
) {
    waits(false)
}

/// [`waiting`]; where it `bursts`, the picker's release call panics.
fn waits(
    bursts: bool,
) -> (
    Bed,
    Arc<Mutex<Calls>>,
    std_mpsc::Receiver<()>,
    std_mpsc::SyncSender<()>,
) {
    let told = Arc::new(Mutex::new(Vec::new()));
    let (reached, at_issue) = std_mpsc::sync_channel(1);
    let (word, go) = std_mpsc::sync_channel(1);
    let (kept, go) = (Arc::clone(&told), Arc::new(Mutex::new(go)));
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("waits", move || {
            Box::new(Waits {
                told: Told(Arc::clone(&kept)),
                reached: reached.clone(),
                go: Arc::clone(&go),
                bursts,
            })
        })
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "waits".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    (bed(Some(spec), true, (2, 8), []), told, at_issue, word)
}

/// The budget is judged behind the lease's issue: what a stage's issue
/// call takes is the decision's time too. A decision that call outlasts
/// leaves as `Busy`, its lease is off the file, and the engine that
/// heard the issue hears the release.
#[tokio::test]
async fn a_decision_its_stages_issue_call_outlasts_leaves_as_busy_and_gives_its_lease_up() {
    let (mut bed, told, at_issue, word) = waiting();
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // The call runs under the turn of the leases' file, on the lease
    // filed ahead of it: no door of the leases answers until it ends.
    // The call took the budget: the next read of the clock, the
    // decision's last, shows the hour past it.
    bed.script.steps.lock().push_back(Step::At(BUDGET));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.live(GATEWAY), 0);
    let (id, late) = (1 << 32 | 1, Some(ReleaseReason::LateDecision));
    assert_eq!(*told.lock(), [(id, None), (id, late)]);
    // Two reads of the clock, as for a decision that leaves.
    assert_eq!(bed.script.readers.lock().len(), 2);
}

/// The fence is judged behind the lease's issue too: a word that closes
/// it while a stage's issue call runs stops the answer, and the lease
/// filed for it is given up as abandoned.
#[tokio::test]
async fn a_fence_closed_by_a_word_inside_a_stages_issue_call_stops_the_answer_and_its_lease() {
    let (mut bed, told, at_issue, word) = waiting();
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // The candidate gives the lead up while the call runs. A successor
    // takes at once on that word: no answer made under the term leaves
    // after it.
    assert!(bed.elector.release().expect("the release"));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(busy(&answer), Reason::NotLeader);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 0);
    assert_eq!(bed.store.live(GATEWAY), 0);
    let (id, abandoned) = (1 << 32 | 1, Some(ReleaseReason::Abandoned));
    assert_eq!(*told.lock(), [(id, None), (id, abandoned)]);
}

/// A lease another hand ended while its decision ran is not the
/// decision's to tell of. A release its gateway hands while the stage's
/// issue call runs waits for the turn that call runs under: the
/// give-up's door applies it first and finds none on file, and the
/// engine hears the issue and the gateway's release alone.
#[tokio::test]
async fn a_lease_another_hand_ended_is_told_nothing_at_the_give_up() {
    let (mut bed, told, at_issue, word) = waiting();
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // Its gateway gives the lease up while the stage's call runs.
    let id = LeaseId(1 << 32 | 1);
    let hand = Hand::new(GATEWAY);
    let gone = ReleaseReason::ClientGone;
    bed.store.hand(&hand, id, Said::Released(None, gone));
    bed.script.steps.lock().push_back(Step::At(BUDGET));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(*told.lock(), [(id.0, None), (id.0, Some(gone))]);
    assert_eq!(bed.store.live(GATEWAY), 0);
}

/// A decider that leaves its work on an empty queue applies what was
/// handed behind its last take, and the engine's lifecycle calls run
/// with it, outside any decision. A stage's call that panics there is
/// held as a decision's fault is, counted apart from the decisions',
/// and the decider stands for its next request.
#[tokio::test]
async fn a_stages_call_that_panics_as_a_decider_leaves_its_work_is_a_fault_it_outlives() {
    let (mut bed, told, at_issue, word) = waits(true);
    let ended = bed.deciders.ended();
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // Its gateway gives the lease up while the stage's call runs: the
    // frame is left to the decider at work, which applies it as it
    // leaves that work.
    let id = LeaseId(1 << 32 | 1);
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    bed.store.hand(&hand, id, gone);
    word.send(()).expect("the stage waits for the case");
    picked(&bed.answer().await);
    let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
    while faults() == 0 && !*ended.borrow() {
        assert!(
            waited.elapsed() < PATIENCE,
            "the decider never left its work"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!*ended.borrow(), "the decider's thread unwound");
    // The call's fault is its own count: no request went unanswered.
    assert_eq!(faults(), 1);
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
    // The lease left the file ahead of the call that panicked.
    assert_eq!(bed.store.live(GATEWAY), 0);
    assert_eq!(*told.lock(), [(id.0, None)]);
    // The decider stands: its next request is decided.
    assert_eq!(bed.submit(frame(2, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    word.send(()).expect("the stage waits for the case");
    picked(&bed.answer().await);
}

/// An answer its stream's queue does not take carries a lease no
/// gateway will name a dispatch for. Its decider hands the release
/// where the queue refuses the answer, full or closed: the lease is
/// off the file and its stage told, abandoned, before the next
/// decision reads a count. An answer the queue took keeps its lease.
#[tokio::test]
async fn a_lease_whose_answer_no_stream_took_is_given_up_where_it_is_dropped() {
    let (mut bed, told, at_issue, word) = waiting();
    let passed = || {
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        word.send(()).expect("the stage waits for the case");
    };
    let job = |bed: &Bed, id: u8, answers: &mpsc::Sender<wire::Decided>| Job {
        request: checked(frame(id, 1)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: answers.clone(),
    };
    // A stream whose queue holds one answer and that reads none: the
    // first answer waits in it, and the second finds it full.
    let (sender, answers) = mpsc::channel(1);
    for id in [1, 2] {
        assert!(bed.deciders.submit(job(&bed, id, &sender)).is_none());
        passed();
    }
    // One decider decides a model's requests in order: it is past the
    // two when it answers a third, on a queue that is read.
    assert_eq!(bed.submit(frame(3, 1)), None);
    passed();
    picked(&bed.answer().await);
    let id = |count: u64| 1 << 32 | count;
    let abandoned = Some(ReleaseReason::Abandoned);
    let mut heard = vec![
        (id(1), None),
        (id(2), None),
        (id(2), abandoned),
        (id(3), None),
    ];
    assert_eq!(*told.lock(), heard);
    assert_eq!(bed.store.live(GATEWAY), 2);
    // The stream ends, and the answer in its queue with it: a fourth
    // answer finds the queue closed.
    drop(answers);
    assert!(bed.deciders.submit(job(&bed, 4, &sender)).is_none());
    passed();
    assert_eq!(bed.submit(frame(5, 1)), None);
    passed();
    picked(&bed.answer().await);
    heard.extend([(id(4), None), (id(4), abandoned), (id(5), None)]);
    assert_eq!(*told.lock(), heard);
    assert_eq!(bed.store.live(GATEWAY), 3);
    let dropped = [&bed.counts.dropped_full, &bed.counts.dropped_gone];
    assert_eq!(dropped.map(|count| count.load(Ordering::Relaxed)), [1, 1]);
}

/// The hand of an unsent answer applies in place where the ring is
/// full, outside any decision. A stage's call that panics there is
/// held as a decision's fault is, counted apart from the decisions',
/// and the decider stands for its next request.
#[tokio::test]
async fn a_stages_panic_at_an_unsent_answers_give_up_is_a_fault_its_decider_outlives() {
    let (mut bed, told, at_issue, word) = waits(true);
    let ended = bed.deciders.ended();
    // A stream that ended: its answer finds the queue closed.
    let (sender, answers) = mpsc::channel(1);
    drop(answers);
    let job = Job {
        request: checked(frame(1, 1)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: sender,
    };
    assert!(bed.deciders.submit(job).is_none());
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // While the stage's call runs, its gateway gives the lease up and
    // fills the ring behind that frame: the decider's own hand finds
    // the ring full and applies in place, where the stage's release
    // call panics.
    let id = LeaseId(1 << 32 | 1);
    let hand = Hand::new(GATEWAY);
    let gone = || Said::Released(None, ReleaseReason::ClientGone);
    bed.store.hand(&hand, id, gone());
    for _ in 1..MemoryStore::HANDOVER {
        bed.store.hand(&hand, LeaseId(0), gone());
    }
    word.send(()).expect("the stage waits for the case");
    let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
    while faults() == 0 && !*ended.borrow() {
        assert!(
            waited.elapsed() < PATIENCE,
            "the decider never gave the lease up"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!*ended.borrow(), "the decider's thread unwound");
    // The call's fault is its own count: no decision panicked.
    assert_eq!(faults(), 1);
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
    // The lease left the file ahead of the call that panicked.
    assert_eq!(bed.store.live(GATEWAY), 0);
    assert_eq!(*told.lock(), [(id.0, None)]);
    // The decider stands: its next request is decided.
    assert_eq!(bed.submit(frame(2, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    word.send(()).expect("the stage waits for the case");
    picked(&bed.answer().await);
}

/// A give-up whose hand applies in place and meets another lease's
/// panicking call there is through: the call is held where the engine
/// is told, and the hand writes its frame behind it. The unsent
/// answer's lease leaves the file at the take behind its hand.
#[tokio::test]
async fn a_give_up_behind_another_leases_panic_at_a_full_ring_is_still_handed() {
    let (mut bed, told, at_issue, word) = waits(true);
    let ended = bed.deciders.ended();
    // The first lease: its answer is taken and the lease stays on file.
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    word.send(()).expect("the stage waits for the case");
    picked(&bed.answer().await);
    assert_eq!(bed.store.live(GATEWAY), 1);
    // The second: a stream that ended, so its answer finds the queue
    // closed.
    let (sender, answers) = mpsc::channel(1);
    drop(answers);
    let job = Job {
        request: checked(frame(2, 1)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: sender,
    };
    assert!(bed.deciders.submit(job).is_none());
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // While the stage's call runs, the gateway gives the first lease up
    // and fills the ring behind that frame.
    let first = LeaseId(1 << 32 | 1);
    let hand = Hand::new(GATEWAY);
    let gone = || Said::Released(None, ReleaseReason::ClientGone);
    bed.store.hand(&hand, first, gone());
    for _ in 1..MemoryStore::HANDOVER {
        bed.store.hand(&hand, LeaseId(0), gone());
    }
    word.send(()).expect("the stage waits for the case");
    // Two held calls: the first lease's release, where the give-up's
    // hand applied in place, and the second lease's own, where the
    // decider takes its turn behind the hand and applies its frame.
    let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
    while faults() < 2 && !*ended.borrow() {
        assert!(
            waited.elapsed() < PATIENCE,
            "the give-up was not handed behind the call that panicked"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!*ended.borrow(), "the decider's thread unwound");
    // No decision panicked: the calls' faults are their own count.
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
    assert_eq!(bed.store.live(GATEWAY), 0);
    assert_eq!(*told.lock(), [(first.0, None), (first.0 + 1, None)]);
}

/// The lease of an answer its stream's queue refuses is off its file
/// when its decider's give-up returns. The decider is at work there,
/// so the hand leaves the frame to it, and its look ahead of a
/// decision passes a held turn: the give-up takes the turn itself.
/// The decider is the case's own, at work on the case's thread. A
/// second thread, which may not apply at that file, reads it.
#[test]
fn a_refused_answers_lease_is_off_its_file_when_its_give_up_returns() {
    let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
    let mut decider = deciding(&bed);
    decider.working = Some(bed.store.at_work());
    let answer = decider.answer(checked(frame(1, 1)), GATEWAY, bed.script.taken);
    let leased = matches!(answer, wire::Decided::Decision { lease: Some(_), .. });
    assert!(leased, "the decision carries a lease");
    let live = || {
        thread::scope(|scope| {
            let read = scope.spawn(|| bed.store.live(GATEWAY));
            read.join().expect("the reading thread")
        })
    };
    assert_eq!(live(), 1);
    // Its stream refuses the answer.
    decider.unsent(GATEWAY, answer);
    assert_eq!(live(), 0, "the release waited for the decider's next take");
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
}

/// A request refused before anything is built applies what is handed
/// too, so a hand at a full ring waits for one request of the file's
/// decider, whatever its answer. The decider here is a former leader
/// behind its closed fence whose gateway still asks it: it refuses each
/// request and never rests, and the hand is through all the same.
#[tokio::test]
async fn a_refused_request_applies_what_a_full_ring_holds() {
    let mut bed = bed(Some(preset("random", 1)), true, (1, 8), []);
    bed.ask(frame(1, 1)).await;
    let hold = || {
        let (reached, held) = std_mpsc::sync_channel(1);
        let (go, word) = std_mpsc::sync_channel(1);
        (Step::Hold { reached, go: word }, held, go)
    };
    // The second request is held at its read of the clock: its decider
    // is at work, and what is handed is left to it.
    let (first, held, go) = hold();
    bed.script.steps.lock().push_back(first);
    assert_eq!(bed.submit(frame(2, 1)), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    // The ring fills, and one hand more waits at it.
    let gone = || Said::Released(None, ReleaseReason::ClientGone);
    let hand = Hand::new(GATEWAY);
    for _ in 0..MemoryStore::HANDOVER {
        bed.store.hand(&hand, LeaseId(0), gone());
    }
    let (through, handed) = std_mpsc::channel();
    let store = Arc::clone(&bed.store);
    let waiting = thread::spawn(move || {
        store.hand(&Hand::new(GATEWAY), LeaseId(0), gone());
        through.send(()).expect("the case waits for the hand");
    });
    // The lead is an hour gone at every read from here on. Two more
    // requests wait behind the held one, and the last is held at its
    // read of the clock: by then the decider has refused two requests
    // and has not rested.
    let hour = Duration::from_secs(3600);
    let (last, held, last_go) = hold();
    *bed.script.shown.lock() = hour;
    bed.script.steps.lock().extend([Step::At(hour), last]);
    assert_eq!(bed.submit(frame(3, 1)), None);
    assert_eq!(bed.submit(frame(4, 1)), None);
    go.send(()).expect("the decider holds the other end");
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock for the last request");
    let handed = handed.recv_timeout(PATIENCE);
    last_go.send(()).expect("the decider holds the other end");
    handed.expect("a refused request applies what the ring holds");
    waiting.join().expect("the hand's thread");
    for id in 2..=4 {
        let answer = bed.answer().await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice()),
            (Reason::NotLeader, &[id][..])
        );
    }
}

/// A lease's dispatch span counts from its decision's first read, and
/// its answer says what is left of the span at the last. A decision
/// inside its budget that leaves under a millisecond of the span has no
/// lease to hand over: it leaves as `Busy`, late, its lease given up.
/// One with no lease has no span to end, and leaves.
#[tokio::test]
async fn a_slow_decisions_lease_says_what_is_left_of_its_span_and_none_left_is_no_answer() {
    // A budget of twice the span: the span ends first.
    let slow = |id: u8| budgeted(frame(id, 1), DISPATCH_TTL * 2);
    let (mut bed, told, at_issue, word) = waiting();
    let id = |count: u64| 1 << 32 | count;
    // The first decision's stage leaves a millisecond of the span.
    let first = DISPATCH_TTL - Duration::from_millis(1);
    assert_eq!(bed.submit(slow(1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    bed.script.steps.lock().push_back(Step::At(first));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(lease_of(&answer).dispatch_ttl_ms, 1);
    let store: &dyn LeasesStore = bed.store.as_ref();
    let filed = store.get(GATEWAY, LeaseId(id(1)));
    let filed = filed.expect("the lease is on file");
    assert_eq!(filed.dispatch_deadline, bed.script.taken + DISPATCH_TTL);
    // The second is taken where the first left the clock, and its
    // stage leaves under a millisecond of the span, half its budget.
    let second = first + DISPATCH_TTL - Duration::from_micros(999);
    assert_eq!(bed.submit_at(slow(2), GATEWAY, first), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    bed.script.steps.lock().push_back(Step::At(second));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
    // A third of the same length, which the wires shed: no lease, so
    // no span to end, and it leaves as its shed.
    let third = second + DISPATCH_TTL - Duration::from_micros(999);
    let reads = [second, third].map(Step::At);
    bed.script.steps.lock().extend(reads);
    let wireless = constrained(slow(3), 0b010, None);
    assert_eq!(bed.submit_at(wireless, GATEWAY, second), None);
    let shed = declined(proto::ShedReason::NoEndpoints, "wires");
    assert_eq!(outcome(&bed.answer().await), &shed);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
    let late = (id(2), Some(ReleaseReason::LateDecision));
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None), late]);
    assert_eq!(bed.store.live(GATEWAY), 1, "the first decision's lease");
}

#[tokio::test]
async fn the_engine_that_decided_is_told_of_a_lease_and_of_its_end_at_the_dispatch_deadline() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    // Two leases at the take, the second dispatched.
    for request in 1..=2 {
        bed.ask(frame(request, 1)).await;
    }
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert!(store.dispatched(
        GATEWAY,
        LeaseId(id(2)),
        Dispatch::default(),
        bed.script.taken
    ));
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    // A decision past the dispatch deadline ends the first as
    // abandoned, and its engine is told; the dispatched lease stands.
    let late = DISPATCH_TTL + Duration::from_secs(1);
    bed.script.steps.lock().push_back(Step::At(late));
    assert_eq!(bed.submit_at(frame(3, 1), GATEWAY, late), None);
    bed.answer().await;
    let abandoned = (id(1), Some(ReleaseReason::Abandoned));
    assert_eq!(told.lock()[2..], [(id(3), None), abandoned]);
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
    assert_eq!(store.live(GATEWAY), 2);
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

/// One lease stands for a request its gateway asks again. The lease
/// the first ask was issued ends as abandoned where it waits, and the
/// engine that issued it is told, ahead of the repeat's own lease. A
/// lease of another request stands.
#[tokio::test]
async fn a_request_asked_again_ends_the_lease_its_first_ask_was_issued() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    for request in 1..=2 {
        bed.ask(frame(request, 1)).await;
    }
    // The answer to the first never arrived: its gateway asks again.
    let answer = bed.ask(again(frame(1, 1))).await;
    assert_eq!(lease_of(&answer).id, id(3));
    let abandoned = (id(1), Some(ReleaseReason::Abandoned));
    assert_eq!(
        *told.lock(),
        [(id(1), None), (id(2), None), abandoned, (id(3), None)]
    );
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
    assert_eq!(store.live(GATEWAY), 2, "another request's and the repeat's");
    assert_eq!(asked_again(&bed), [0, 1]);
}

/// A request sent twice under one id without the word is two requests:
/// each is issued a lease and neither ends the other's.
#[tokio::test]
async fn a_request_sent_twice_without_the_word_is_issued_two_leases() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    for _ in 0..2 {
        bed.ask(frame(1, 1)).await;
    }
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    assert_eq!(bed.store.live(GATEWAY), 2);
    assert_eq!(asked_again(&bed), [0, 0]);
}

/// A repeat ends a lease that waits and no other: a lease its gateway
/// named a dispatch for stands, and a request no decider took has
/// none. Each such repeat is decided as a first ask.
#[tokio::test]
async fn a_request_asked_again_ends_no_lease_that_was_dispatched() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    let store: &dyn LeasesStore = bed.store.as_ref();
    let first = LeaseId(id(1));
    assert!(store.dispatched(GATEWAY, first, Dispatch::default(), bed.script.taken));
    // The gateway holds the first answer and asks again all the same,
    // and it asks again for a request no decider ever took.
    for request in [1, 9] {
        bed.ask(again(frame(request, 1))).await;
    }
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None), (id(3), None)]);
    assert_eq!(bed.store.live(GATEWAY), 3);
    assert_eq!(asked_again(&bed), [2, 0]);
}

/// The word reaches the asking gateway's own leases alone: a lease
/// another gateway holds under the same id stands.
#[tokio::test]
async fn a_request_asked_again_ends_no_lease_of_another_gateway() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    let repeat = again(frame(1, 1));
    assert_eq!(bed.submit_at(repeat, OTHER, Duration::ZERO), None);
    bed.answer().await;
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    assert_eq!((bed.store.live(GATEWAY), bed.store.live(OTHER)), (1, 1));
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// A repeat ends its earlier lease whatever becomes of its own
/// decision: one answered `Busy` for a budget that is gone leaves no
/// lease of its request on file.
#[tokio::test]
async fn a_request_asked_again_past_its_budget_ends_its_earlier_lease_all_the_same() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    // The repeat's budget is gone at its decider's read of the clock,
    // well inside the first lease's dispatch span.
    let (budget, read) = (Duration::from_millis(1), Duration::from_millis(2));
    bed.script.steps.lock().push_back(Step::At(read));
    let answer = bed.ask(budgeted(again(frame(1, 1)), budget)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    let abandoned = (id(1), Some(ReleaseReason::Abandoned));
    assert_eq!(*told.lock(), [(id(1), None), abandoned]);
    assert_eq!(bed.store.live(GATEWAY), 0);
    assert_eq!(asked_again(&bed), [0, 1]);
}

/// A dispatch handed ahead of the repeat stands: what was handed to
/// the file is applied before the repeat looks for a lease that waits.
#[tokio::test]
async fn a_dispatch_handed_ahead_of_a_request_asked_again_is_applied_first() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (1, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    // The repeat's decider is held at its first read of the clock: it
    // has the job at hand, so a frame handed now is left to it.
    let (reached, held) = std_mpsc::sync_channel(1);
    let (go, word) = std_mpsc::sync_channel(1);
    let hold = Step::Hold { reached, go: word };
    bed.script.steps.lock().push_back(hold);
    assert_eq!(bed.submit(again(frame(1, 1))), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    let hand = Hand::new(GATEWAY);
    let dispatched = Said::Dispatched(Dispatch::default(), bed.script.taken);
    bed.store.hand(&hand, LeaseId(id(1)), dispatched);
    go.send(()).expect("the decider holds the other end");
    bed.answer().await;
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    assert_eq!(
        bed.store.live(GATEWAY),
        2,
        "the dispatched lease and the repeat's"
    );
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// A decider with a job at hand is at work at the leases' file: what a
/// session hands meanwhile is left to it, and it applies it before its
/// decision reads the rows' counts, ahead of its own issue. A frame
/// handed while no decider has a job is applied all the same.
#[tokio::test]
async fn a_frame_handed_while_a_decider_decides_is_applied_by_that_decider() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&told);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("told", move || Box::new(Told(Arc::clone(&kept))))
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "told".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    let mut bed = bed(Some(spec), true, (1, 8), []);
    let id = |count: u64| 1 << 32 | count;
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    bed.ask(frame(1, 1)).await;
    // The second decision is held at its first read of the clock: its
    // decider has the job at hand.
    let (reached, held) = std_mpsc::sync_channel(1);
    let (go, word) = std_mpsc::sync_channel(1);
    let hold = Step::Hold { reached, go: word };
    bed.script.steps.lock().push_back(hold);
    assert_eq!(bed.submit(frame(2, 1)), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    let hand = Hand::new(GATEWAY);
    bed.store.hand(&hand, LeaseId(id(1)), gone);
    assert_eq!(*told.lock(), [(id(1), None)], "left to the decider at work");
    go.send(()).expect("the decider holds the other end");
    bed.answer().await;
    let released = |count| (id(count), Some(ReleaseReason::ClientGone));
    assert_eq!(*told.lock(), [(id(1), None), released(1), (id(2), None)]);
    // The decider waits on its empty queue, or is about to: it applies
    // as it leaves its work, or the hand does.
    bed.store.hand(&hand, LeaseId(id(2)), gone);
    settled(|| told.lock().len() >= 4).await;
    assert_eq!(told.lock()[3], released(2));
}

/// The `random` picker, keeping the decider its thread acts as at each
/// call of a lease's issue and release.
#[derive(Debug)]
struct Named(Arc<Mutex<Vec<Option<DecisionSlot>>>>);

impl Lifecycle for Named {
    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        self.0.lock().push(DecisionSlot::acting());
    }

    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        self.0.lock().push(DecisionSlot::acting());
    }
}

impl Picker for Named {
    fn name(&self) -> &'static str {
        "named"
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

/// A decider's thread acts as its decider while it decides, and a
/// lease's event is told under the name of the decider whose file holds
/// the lease, whichever thread applies the frame.
#[tokio::test]
async fn a_decision_and_its_leases_events_act_as_the_leases_decider() {
    let named = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&named);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("named", move || Box::new(Named(Arc::clone(&kept))))
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "named".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    let mut bed = bed(Some(spec), true, (1, 8), []);
    let answer = bed.ask(frame(1, 1)).await;
    // The case's thread names no decider: its hand applies the release,
    // or the decider does as it leaves its work.
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    bed.store.hand(&hand, LeaseId(lease_of(&answer).id), gone);
    settled(|| named.lock().len() >= 2).await;
    assert_eq!(*named.lock(), [Some(DecisionSlot::new(0)); 2]);
    assert_eq!(DecisionSlot::acting(), None);
}

/// The `random` picker, keeping the name of the thread each release of
/// a lease is told on.
#[derive(Debug)]
struct Threads(Arc<Mutex<Vec<Option<String>>>>);

impl Lifecycle for Threads {
    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        let name = thread::current().name().map(str::to_owned);
        self.0.lock().push(name);
    }
}

impl Picker for Threads {
    fn name(&self) -> &'static str {
        "threads"
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

/// A decider that waits for a job with a lease on its file keeps its
/// watch: a frame handed on the lull is left in the ring, and the
/// decider's own thread applies it within its look. The hand's thread
/// applies none. Once no lease stands on the file the watch ends, and
/// the decider wakes for a job alone.
#[tokio::test]
async fn a_frame_handed_on_a_lull_is_applied_by_its_leases_decider() {
    let threads = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&threads);
    let spec = picking("threads", move || Box::new(Threads(Arc::clone(&kept))));
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let taker = bed.deciders.of(MODEL);
    let answers = [bed.ask(frame(1, 1)).await, bed.ask(frame(2, 1)).await];
    // Two leases stand on its file: its look behind the answers
    // starts its watch, and it waits for a job from there.
    watching(&bed, 1).await;
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    for answer in &answers {
        bed.store.hand(&hand, LeaseId(lease_of(answer).id), gone);
    }
    settled(|| threads.lock().len() >= 2).await;
    let decider = || Some(format!("decide-{taker}"));
    assert_eq!(*threads.lock(), [decider(), decider()]);
    // No lease stands on the file: the decider's next look ends its
    // watch.
    watching(&bed, 0).await;
}

/// A controller's only decider keeps no watch. With leases on its
/// file and no job it rests, so a frame handed on the lull is applied
/// before its hand returns: by the hand in place, or by the decider's
/// last take where the hand meets its rest.
#[tokio::test]
async fn a_controllers_only_decider_keeps_no_watch() {
    let threads = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&threads);
    let spec = picking("threads", move || Box::new(Threads(Arc::clone(&kept))));
    let mut bed = bed(Some(spec), true, (1, 8), []);
    let answers = [bed.ask(frame(1, 1)).await, bed.ask(frame(2, 1)).await];
    // The decider leaves its work behind its last answer: the case
    // waits until the store shows no thread at work.
    settled(|| bed.store.working() == 0).await;
    assert_eq!(bed.store.watches(), 0, "two leases stand on its file");
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    for (applied, answer) in answers.iter().enumerate() {
        bed.store.hand(&hand, LeaseId(lease_of(answer).id), gone);
        let told = threads.lock().len();
        assert_eq!(told, applied + 1, "applied before its hand returned");
    }
    assert_eq!(bed.store.watches(), 0, "no look is owed");
}

/// Waits until `files` of the bed's files show a watch. A decider
/// starts its watch at its look, behind its answer, and ends it at a
/// look: a read right behind an answer may come first.
async fn watching(bed: &Bed, files: usize) {
    settled(|| bed.store.watches() == files).await;
}

/// A decider that waits with a lease on file waits its floor behind a
/// job, twice as long behind each look that applied no frame, up to
/// its cap, and its floor again behind a look that applied one: a
/// file whose frames flow is not quiet. Once no lease stands on the
/// file it waits with no bound. The decider is the case's own, and
/// the case's receive answers each wait as a queue with no job would.
#[test]
fn a_watching_deciders_wait_doubles_over_a_quiet_file_and_falls_at_a_frame() {
    let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
    let mut decider = deciding(&bed);
    decider.working = Some(bed.store.at_work());
    let answer = decider.answer(checked(frame(1, 1)), GATEWAY, bed.script.taken);
    let lease = LeaseId(lease_of(&answer.frame()).id);
    let hand = Hand::new(GATEWAY);
    let mut waits = Vec::new();
    let job = decider.wait_by(|within| {
        waits.push(within.map(|period| period.as_millis()));
        match waits.len() {
            // Handed in the seventh wait: the look behind it applies
            // a frame.
            7 => {
                let dispatch = Said::Dispatched(Dispatch::default(), bed.script.taken);
                bed.store.hand(&hand, lease, dispatch);
            }
            // Handed in the ninth: the look behind it applies the
            // release of the file's last lease.
            9 => {
                let gone = Said::Released(None, ReleaseReason::ClientGone);
                bed.store.hand(&hand, lease, gone);
            }
            10 => return Err(std_mpsc::RecvTimeoutError::Disconnected),
            _ => {}
        }
        Err(std_mpsc::RecvTimeoutError::Timeout)
    });
    assert!(job.is_none(), "the case's queue ends with no job");
    // Seven waits over the quiet file, two behind the frame, and no
    // bound behind the release.
    let mut wanted: Vec<_> = [1, 2, 4, 8, 16, 16, 16, 1, 2].map(Some).into();
    wanted.push(None);
    assert_eq!(waits, wanted);
    assert_eq!((LOOK_FLOOR.as_millis(), LOOK_CAP.as_millis()), (1, 16));
}

/// A decider with no place yet keeps no watch: before its first job
/// its wait is handed no period, as over a file that holds no lease.
#[test]
fn a_decider_with_no_place_yet_waits_with_no_bound() {
    let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
    let mut decider = deciding(&bed);
    let mut waits = Vec::new();
    let job = decider.wait_by(|within| {
        waits.push(within);
        Err(std_mpsc::RecvTimeoutError::Disconnected)
    });
    assert!(job.is_none(), "the case's queue ends with no job");
    assert_eq!(waits, [None], "no place, so no watch and no period");
    assert_eq!(bed.store.watches(), 0);
}

/// The `random` picker, keeping the instant each release of a lease is
/// told at.
#[derive(Debug)]
struct Heard(Arc<Mutex<Vec<Instant>>>);

impl Lifecycle for Heard {
    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        self.0.lock().push(Instant::now());
    }
}

impl Picker for Heard {
    fn name(&self) -> &'static str {
        "heard"
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

/// A decision under a stage that may refuse by a report of leases
/// reads a completion its gateway reported of a lease on another
/// decider's file. That decider waits for a job with its watch over
/// the file, so the frames were left in its ring; the deciding thread
/// applies them before it reads its rows. The two models are taken by
/// two deciders and served by the same members.
#[tokio::test]
async fn a_decision_whose_stage_may_refuse_by_reports_applies_what_waits_at_the_other_files() {
    use crate::service::controller::lifecycle::tests::{a_completed, a_dispatched};

    let rows = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(seeing(&rows, Counted)), true, (2, 8), []);
    let models = a_model_each(&bed.deciders, 2);
    let served: Vec<&str> = models.iter().map(String::as_str).collect();
    for name in ["s0", "s1", "s2"] {
        serving(
            &bed.store,
            &bed.reads.objects,
            name,
            &served,
            Role::Single,
            WireModes::NONE,
        );
    }
    let hand = Hand::new(GATEWAY);
    let answer = bed.ask(frame_for(&models[0], 1, 1)).await;
    let lease = lease_of(&answer).id;
    // The first model's decider waits for a job on its watch over the
    // lease.
    watching(&bed, 1).await;
    let at = Instant::now();
    lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease), at).expect("the frame reads");
    lifecycle::completed(&bed.store, &hand, &a_completed(lease, picked(&answer)))
        .expect("the frame reads");
    // The other model's decision, at the other file, right behind the
    // completion: it reads no request in flight.
    bed.ask(frame_for(&models[1], 2, 1)).await;
    let rows = rows.lock();
    let read: u32 = rows[1].iter().map(|(count, _)| count).sum();
    assert_eq!(read, 0, "{rows:?}");
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

/// The idle cost of a watch, for its measurement at `deciders`
/// deciders: one dispatched lease on each decider's file and no job,
/// for five seconds between two printed lines. A reader of the
/// threads named `decide-*` counts their wakes and their time on a
/// core over the pause. The tenth of a second before the first line
/// is not counted: each decider's wait grows to its cap in it.
async fn idle(deciders: usize) {
    use std::io::Write as _;

    use crate::service::controller::lifecycle::tests::a_dispatched;

    let mut out = io::stdout();
    let mut bed = bed(Some(preset("random", 1)), true, (deciders, 8), []);
    let models = a_model_each(&bed.deciders, deciders);
    let served: Vec<&str> = models.iter().map(String::as_str).collect();
    for name in ["s0", "s1", "s2"] {
        serving(
            &bed.store,
            &bed.reads.objects,
            name,
            &served,
            Role::Single,
            WireModes::NONE,
        );
    }
    let hand = Hand::new(GATEWAY);
    for model in &models {
        let answer = bed.ask(frame_for(model, 1, 1)).await;
        let dispatched = a_dispatched(lease_of(&answer).id);
        lifecycle::dispatched(&bed.store, &hand, &dispatched, Instant::now())
            .expect("the frame reads");
    }
    thread::sleep(Duration::from_millis(100));
    writeln!(
        out,
        "the idle pause begins: a lease on each of {deciders} files, no job"
    )
    .expect("stdout");
    thread::sleep(Duration::from_secs(5));
    writeln!(out, "the idle pause ends").expect("stdout");
}

/// The idle cost of one decider's watch ([`idle`]).
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn one_idle_decider_measured() {
    idle(1).await;
}

/// The idle cost of eight deciders' watches ([`idle`]).
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn eight_idle_deciders_measured() {
    idle(8).await;
}

/// The lag of a frame handed on a lull, measured at one of two
/// deciders: from its hand to the decider's tell, over `FRAMES`
/// frames a cell. At
/// the floor each frame is handed within one shortest wait of its
/// decision. At the cap it is handed three longest waits behind its
/// decision and within one more: the decider's wait has grown to its
/// cap by then. Each is handed at a phase that sweeps that wait, so
/// no frame's instant is locked to a look.
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_lulls_lag_measured() {
    use std::io::Write as _;

    const FRAMES: usize = 500;
    let mut out = io::stdout();
    let heard = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&heard);
    let spec = picking("heard", move || Box::new(Heard(Arc::clone(&kept))));
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    let parts = u32::try_from(FRAMES).expect("five hundred");
    let cells = [
        ("at the floor", Duration::ZERO, LOOK_FLOOR),
        ("at the cap", 3 * LOOK_CAP, LOOK_CAP),
    ];
    let mut told = 0;
    for (cell, quiet, wait) in cells {
        let mut lags = Vec::with_capacity(FRAMES);
        for sent in 0..FRAMES {
            let answer = bed.ask(frame(1, 1)).await;
            // A step coprime to the sweep: each part of the wait once.
            let part = u32::try_from(sent * 7_919 % FRAMES).expect("under five hundred");
            thread::sleep(quiet + wait * part / parts);
            let handed = Instant::now();
            bed.store.hand(&hand, LeaseId(lease_of(&answer).id), gone);
            let at = loop {
                if let Some(at) = heard.lock().get(told) {
                    break *at;
                }
                assert!(
                    handed.elapsed() < PATIENCE,
                    "a handed frame waits one look at most"
                );
                std::hint::spin_loop();
            };
            told += 1;
            lags.push(at.saturating_duration_since(handed));
        }
        lags.sort_unstable();
        let at = |part: usize| lags[(FRAMES * part / 100).min(FRAMES - 1)].as_micros();
        writeln!(
                out,
                "the lag of a frame handed on a lull {cell}, {FRAMES} frames: p50 {} us, p99 {} us, the longest {} us",
                at(50),
                at(99),
                at(100)
            )
            .expect("stdout");
    }
}

/// One closed-loop client, measured: the share of its decisions that
/// read a count from before its last completion. The client asks,
/// names its dispatch, reports the completion `served` later and asks
/// again `think` later, so no lease of its own stands at a decision:
/// one that reads a request in flight read a count from before the
/// completion. Its requests are for one model, or for two in turn
/// that two deciders take and the same members serve: each decision
/// of the second form is made at another file than the completion
/// before it.
/// The stage that reads the counts is made by `stage`, and `under`
/// names it in the cell's lines.
async fn closed_loop<P: Picker + 'static>(
    stage: impl Fn(Seen) -> P + Send + Sync + 'static,
    under: &str,
) {
    use std::io::Write as _;

    use crate::service::controller::lifecycle::tests::{a_completed, a_dispatched};

    const ASKED: usize = 200;
    let mut out = io::stdout();
    let rows = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(seeing(&rows, stage)), true, (2, 8), []);
    let models = a_model_each(&bed.deciders, 2);
    let served: Vec<&str> = models.iter().map(String::as_str).collect();
    for name in ["s0", "s1", "s2"] {
        serving(
            &bed.store,
            &bed.reads.objects,
            name,
            &served,
            Role::Single,
            WireModes::NONE,
        );
    }
    let hand = Hand::new(GATEWAY);
    let pause = |long: Duration| {
        if !long.is_zero() {
            thread::sleep(long);
        }
    };
    let millis = Duration::from_millis;
    let paces = [
        (millis(0), millis(0)),
        (millis(0), millis(2)),
        (millis(48), millis(0)),
        (millis(48), millis(20)),
    ];
    for (form, turn) in [("one model", [0, 0]), ("two models in turn", [0, 1])] {
        for (served, think) in paces {
            rows.lock().clear();
            for asked in 0..ASKED {
                let answer = bed.ask(frame_for(&models[turn[asked % 2]], 1, 1)).await;
                let lease = lease_of(&answer).id;
                lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease), Instant::now())
                    .expect("the frame reads");
                pause(served);
                let completed = a_completed(lease, picked(&answer));
                lifecycle::completed(&bed.store, &hand, &completed).expect("the frame reads");
                pause(think);
            }
            // The first decision of a cell stands behind the cell
            // before it: it is not counted.
            let rows = rows.lock();
            let behind = rows.iter().skip(1);
            let late = behind.filter(|read| read.iter().any(|(count, _)| *count > 0));
            writeln!(
                    out,
                    "a closed loop of {form}{under}, served {} ms, think {} ms: {} of {} decisions read a count from before the last completion",
                    served.as_millis(),
                    think.as_millis(),
                    late.count(),
                    ASKED - 1
                )
                .expect("stdout");
        }
    }
}

/// [`closed_loop`] under a stage that only picks by its counts.
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_closed_loop_measured() {
    closed_loop(|seen| seen, "").await;
}

/// [`closed_loop`] under a stage that says it may refuse by a report
/// of leases.
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_closed_loop_under_a_counting_stage_measured() {
    closed_loop(Counted, " under a stage that may refuse by a report").await;
}

/// An abandoned lease is told to the engine whose generation issued it
/// and to no stage a later one built new: such a stage never saw the
/// issue. Each generation here is validated over a stage table of its
/// own, so the later one keeps none. The lease is ended all the same.
#[tokio::test]
async fn a_later_generations_engine_is_not_told_of_a_lease_it_did_not_issue() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let spec = |generation: u64| hearing(&told, generation);
    let mut bed = bed(Some(spec(1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    // The next generation stands when the first lease's deadline is
    // past: its engine is told of its own issue and of nothing else.
    bed.store.apply(spec(2)).expect("a later generation");
    let late = DISPATCH_TTL + Duration::from_secs(1);
    bed.script.steps.lock().push_back(Step::At(late));
    assert_eq!(bed.submit_at(frame(2, 1), GATEWAY, late), None);
    bed.answer().await;
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
    assert_eq!(store.live(GATEWAY), 1);
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

/// A decision reads what the leases on file book, whatever its profile
/// and with nothing declared: the three leases the decisions before it
/// left are counted on their workers' objects, and its records carry
/// each worker's count.
#[tokio::test]
async fn a_decision_reads_the_leases_on_file_in_its_workers_records() {
    let rows = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&rows);
    let mut catalog = StageRegistry::builtin();
    let seen = move || -> Box<dyn Picker> {
        let rows = Arc::clone(&kept);
        Box::new(Seen { rows })
    };
    catalog.picker("seen", seen).expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "seen".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    // Three decisions and no word from their gateway: three leases
    // stand on file when the fourth is made.
    for request in 1..=4u8 {
        bed.ask(frame(request, u64::from(request))).await;
    }
    // What the fourth decision was handed: its facts read the three
    // leases, in its records.
    let rows = rows.lock();
    let (read, own): (Vec<u32>, Vec<u32>) = rows[3].iter().copied().unzip();
    assert_eq!(read.iter().sum::<u32>(), 3, "{rows:?}");
    assert_eq!(own.iter().sum::<u32>(), 3, "{rows:?}");
}

/// A decision can pick a worker the store's world does not list yet:
/// the worker objects follow a registration as its write lands, and
/// the store builds the pools that list it a window later. The
/// lease of such a pick books the worker's count all the same, by the
/// slot the registration filed: the next decision reads one request in
/// flight on the worker, on its object and in its record. A lease that
/// booked nothing would leave every decision of the window reading the
/// new worker idle.
#[tokio::test]
async fn a_pick_inside_its_workers_registration_window_books_its_count() {
    const LATE: &str = "model-late";
    let rows = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&rows);
    let spec = picking("seen", move || -> Box<dyn Picker> {
        let rows = Arc::clone(&kept);
        Box::new(Seen { rows })
    });
    let mut bed = bed(Some(spec), true, (1, 8), []);
    // One more member, alone under its model, written ready and never
    // published: the build that lists it is still owed.
    let key = String::from("http://late:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role: Role::Single,
        models: vec![LATE.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    let late = rewritten(&bed, facts);
    let row = WorkersStore::get(&*bed.store, late).expect("the row");
    bed.reads
        .objects
        .connect_row(&bed.store, late, row.revision);
    let listed = bed.store.snapshot().booked_of(late);
    assert_eq!(listed, None, "no world lists the worker yet");
    for request in 1..=2u8 {
        let answer = bed.ask(frame_for(LATE, request, u64::from(request))).await;
        assert_eq!(picked(&answer), late);
    }
    // What the second decision was handed: the first one's lease, in
    // the count its facts read and in the worker's record.
    assert_eq!(rows.lock()[1], [(1, 1)]);
}

/// A frame with under a microsecond of its budget left where its
/// decider takes it is answered there, and no request is made of it:
/// no stage is opened. A request that stated a budget of 0 would say
/// that its frame states none.
#[tokio::test]
async fn a_frame_with_nothing_left_is_answered_and_no_stage_is_opened() {
    let rows = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&rows);
    let spec = picking("seen", move || -> Box<dyn Picker> {
        let rows = Arc::clone(&kept);
        Box::new(Seen { rows })
    });
    let spent = BUDGET - Duration::from_nanos(999);
    let steps = [spent, Duration::ZERO].map(Step::At);
    let mut bed = bed(Some(spec), true, (1, 8), steps);
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert!(rows.lock().is_empty(), "no stage was opened");
    // The next frame is taken with its whole budget and is decided.
    let answer = bed.ask(frame(2, 2)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(rows.lock().len(), 1);
}

#[tokio::test]
async fn a_request_whose_deciders_queue_is_full_comes_back_at_once() {
    let (reached, at_read) = std_mpsc::sync_channel(1);
    let (release, go) = std_mpsc::sync_channel(1);
    // No decider and no depth asked: one thread behind a queue of one.
    let spec = Some(preset("random", 1));
    let mut bed = bed(spec, true, (0, 0), [Step::Hold { reached, go }]);
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_read
        .recv_timeout(PATIENCE)
        .expect("the decider took the first request");
    assert_eq!(bed.submit(frame(2, 2)), None, "the queue holds one");
    assert_eq!(bed.submit(frame(3, 3)), Some(Refused::Full));
    release.send(()).expect("the decider waits for the case");
    // The two that took a place are decided in the order they were
    // handed over; the third never reached the decider.
    for id in [1, 2] {
        let answer = bed.answer().await;
        assert_eq!(answer.request_id, [id]);
        assert!(bed.workers.contains(&picked(&answer)));
    }
    assert_eq!(bed.ask(frame(4, 4)).await.request_id, [4]);
}

#[tokio::test]
async fn an_answer_its_streams_full_queue_does_not_take_is_dropped_and_the_decider_goes_on() {
    let mut bed = leading([]);
    // A stream whose queue holds two answers and that reads none.
    let (sender, mut answers) = mpsc::channel(2);
    for id in 1..=3 {
        let job = Job {
            request: checked(frame(id, 1)),
            gateway: GATEWAY,
            received: bed.script.taken,
            answers: sender.clone(),
        };
        assert!(bed.deciders.submit(job).is_none());
    }
    // One decider decides a model's requests in order: it is past the
    // three when it answers a fourth, on a queue that is read.
    assert_eq!(bed.ask(frame(4, 1)).await.request_id, [4]);
    for id in [1, 2] {
        let answer = answers.recv().await.expect("the queue held it");
        assert_eq!(answer.frame().request_id, [id]);
    }
    assert!(
        answers.try_recv().is_err(),
        "the third found the queue full"
    );
    let dropped = |counts: &DecideStats| {
        [&counts.dropped_full, &counts.dropped_gone].map(|count| count.load(Ordering::Relaxed))
    };
    assert_eq!(dropped(&bed.counts), [1, 0]);
    // A stream that ended takes none: its answer is counted as gone.
    drop(answers);
    let job = Job {
        request: checked(frame(5, 1)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: sender,
    };
    assert!(bed.deciders.submit(job).is_none());
    assert_eq!(bed.ask(frame(6, 1)).await.request_id, [6]);
    assert_eq!(dropped(&bed.counts), [1, 1]);
}

/// Each `Busy` a decider answers is counted at its reason, and a
/// decision moves no count.
#[tokio::test]
async fn each_busy_is_counted_at_its_reason_and_a_decision_moves_no_count() {
    // A decision; a request taken with its budget gone; one whose
    // decision panics; two taken past the renew deadline.
    let steps = [
        Step::At(Duration::ZERO),
        Step::At(Duration::ZERO),
        Step::At(BUDGET),
        Step::Panic,
        Step::At(FENCED),
    ];
    let mut bed = bed(Some(preset("random", 1)), true, (1, 8), steps);
    let counted = |counts: &DecideStats| {
        counts
            .busy
            .each_ref()
            .map(|count| count.load(Ordering::Relaxed))
    };
    let answer = bed.ask(frame(1, 1)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(counted(&bed.counts), [0; 9]);
    assert_eq!(
        busy(&bed.ask(frame(2, 2)).await),
        Reason::BudgetGoneOnArrival
    );
    assert_eq!(busy(&bed.ask(frame(3, 3)).await), Reason::Unspecified);
    for id in [4, 5] {
        assert_eq!(busy(&bed.ask(frame(id, 1)).await), Reason::NotLeader);
    }
    // In the order of the reasons' numbers on the wire.
    assert_eq!(
        DecideStats::BUSY[3..5],
        ["budget_gone_on_arrival", "not_leader"]
    );
    assert_eq!(counted(&bed.counts), [1, 0, 0, 1, 2, 0, 0, 0, 0]);
    // The counts are sized by hand beside the wire's enum: every reason
    // the contract names has a place, and the number past them is no
    // reason. A reason added to the contract fails here, not silently.
    let places = i32::try_from(DecideStats::BUSY.len()).expect("a handful of reasons");
    assert!((0..places).all(|number| Reason::try_from(number).is_ok()));
    assert!(Reason::try_from(places).is_err());
}

#[tokio::test]
async fn a_budget_the_queue_used_up_is_busy_and_one_microsecond_of_it_is_a_decision() {
    let short = BUDGET - Duration::from_micros(1);
    // The first request is taken with one microsecond left, the second
    // with none, the third with under one: the engine reads what is
    // left in whole microseconds, and 0 as none.
    let under = BUDGET - Duration::from_nanos(999);
    let steps = [short, short, BUDGET, under].map(Step::At);
    let mut bed = leading(steps);
    let answer = bed.ask(frame(1, 1)).await;
    assert!(bed.workers.contains(&picked(&answer)));
    let answer = bed.ask(frame(2, 2)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!((answer.request_id.as_slice(), answer.term), (&[2][..], 1));
    let answer = bed.ask(frame(3, 3)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    // One read of the clock each: nothing was read for either after
    // the budget.
    assert_eq!(bed.script.readers.lock().len(), 4);
}

#[tokio::test]
async fn a_replica_that_does_not_lead_answers_not_leader_under_the_highest_term_it_knows() {
    // The third request is taken a whole budget after its receipt.
    let steps = [Duration::ZERO, Duration::ZERO, BUDGET].map(Step::At);
    let mut bed = bed(Some(preset("random", 1)), false, (2, 8), steps);
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 0));
    // The election saw another replica hold the lease under term 4.
    bed.standing.send_replace(ElectionObservation {
        holder: "controller-1@10.0.0.8:30100".to_owned(),
        term: 4,
        looked: true,
    });
    let answer = bed.ask(frame(2, 2)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 4));
    // The fence is read before the budget: the reason that still holds
    // for the next request is the one said.
    let answer = bed.ask(frame(3, 3)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 4));
    // One read of the clock each: nothing was built for any.
    assert_eq!(bed.script.readers.lock().len(), 3);
}

#[tokio::test]
async fn a_leader_past_its_renew_deadline_answers_not_leader_under_its_own_term() {
    let mut bed = leading([Step::At(FENCED)]);
    assert_eq!(bed.submit_at(frame(1, 1), GATEWAY, FENCED), None);
    let answer = bed.answer().await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
}

#[tokio::test]
async fn a_fence_that_closes_between_the_build_and_the_hand_over_sends_busy_in_the_decisions_place()
{
    // The first read finds the fence open and the decision is made; the
    // read at the hand-over is past the renew deadline.
    let mut bed = leading([Step::At(Duration::ZERO), Step::At(FENCED)]);
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.generation = 2;
    document.explain.sample = 1;
    bed.store
        .apply(applied(&document))
        .expect("a later generation");
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
    assert_eq!(bed.script.readers.lock().len(), 2);
    assert!(bed.deciders.explain(1, GATEWAY, &[1]).is_none());
    assert!(bed.deciders.shared.samples.get().is_none());
}

#[tokio::test]
async fn an_unsampled_served_decision_lays_out_no_rings() {
    let mut bed = leading([]);
    let answer = bed.ask(frame(1, 1)).await;
    assert!(matches!(outcome(&answer), Outcome::Pick(_)));
    assert!(bed.deciders.explain(1, GATEWAY, &[1]).is_none());
    assert!(bed.deciders.shared.samples.get().is_none());
    assert_eq!(bed.script.readers.lock().len(), 2);
}

/// A served decision that is kept is filed under the instant its
/// decider read its clock at, which is not the instant its frame was
/// received.
#[tokio::test]
async fn a_kept_served_decision_is_filed_under_its_deciders_instant() {
    // Every read of the case's clock shows the take plus this much.
    // The frame is received at the take.
    const LATER: Duration = Duration::from_millis(250);
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.generation = 1;
    document.explain.sample = 1;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), [Step::At(LATER)]);
    let answer = bed.ask(frame(1, 1)).await;
    assert!(matches!(outcome(&answer), Outcome::Pick(_)));
    let record = bed.deciders.explain(answer.term, GATEWAY, &[1]);
    let record = record.expect("the kept decision");
    assert_eq!(record.at, bed.script.taken + LATER);
}

#[test]
fn a_served_decision_that_keeps_no_record_answers_during_a_read() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.generation = 1;
    document.explain.sample = 2;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("the case's runtime");
    for id in 1..=2 {
        runtime.block_on(bed.ask(frame(id, 1)));
    }

    let shared = Arc::clone(&bed.deciders.shared);
    let rings = shared.samples.get().expect("the ring of the first sample");
    let _read = rings[0].lock();
    let answer = runtime.block_on(bed.ask(frame(3, 1)));
    assert!(matches!(outcome(&answer), Outcome::Pick(_)));
}

#[tokio::test]
async fn a_new_term_restarts_the_served_sample_count() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.generation = 1;
    document.explain.sample = 2;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let first = bed.ask(frame(1, 1)).await;
    assert!(matches!(outcome(&first), Outcome::Pick(_)));

    assert!(bed.elector.release().expect("the release"));
    assert_eq!(
        bed.elector.tick(bed.script.taken).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    bed.store.set_term(2);
    let answer = bed.ask(frame(2, 1)).await;
    assert_eq!(answer.term, 2);
    assert!(matches!(outcome(&answer), Outcome::Pick(_)));
    assert!(bed.deciders.explain(2, GATEWAY, &[2]).is_none());
}

#[tokio::test]
async fn a_fence_open_under_a_later_term_at_the_hand_over_sends_busy_too() {
    let (reached, at_read) = std_mpsc::sync_channel(1);
    let (release, go) = std_mpsc::sync_channel(1);
    let mut bed = leading([Step::At(Duration::ZERO), Step::Hold { reached, go }]);
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_read
        .recv_timeout(PATIENCE)
        .expect("the decider reached the hand-over");
    // The decision was made under term 1. The candidate gives the lease
    // up and takes it again: the fence is open, under term 2.
    assert!(bed.elector.release().expect("the release"));
    let tick = bed.elector.tick(bed.script.taken).expect("the take");
    assert_eq!(tick, TickOutcome::Acquired { term: 2 });
    // The election's report tells the store the term of each take.
    bed.store.set_term(2);
    release.send(()).expect("the decider waits for the case");
    let answer = bed.answer().await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 2));
    // The next request is decided under the term that stands.
    let answer = bed.ask(frame(2, 2)).await;
    assert_eq!(answer.term, 2);
    assert!(bed.workers.contains(&picked(&answer)));
}

/// The term a decision leaves under is the fence's, the one both reads
/// showed. A higher one the election has seen since is a `Busy`'s to
/// carry.
#[tokio::test]
async fn a_decision_leaves_under_the_fences_term_while_the_election_has_seen_a_later_one() {
    let mut bed = leading([]);
    bed.standing.send_replace(ElectionObservation {
        holder: "controller-1@10.0.0.8:30100".to_owned(),
        term: 4,
        looked: true,
    });
    let answer = bed.ask(frame(1, 1)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(answer.term, 1);
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

#[tokio::test]
async fn a_decision_is_made_for_the_gateway_its_request_came_from() {
    let mut bed = leading([]);
    let vetoed = bed.workers[0];
    let until = bed.script.taken + Duration::from_secs(60);
    bed.store.gateway_veto(GATEWAY, vetoed, Some(until));
    let mut theirs = Vec::new();
    for (id, seed) in (1..=48).zip(1..) {
        let own = bed.ask(frame(id, seed)).await;
        assert_ne!(picked(&own), vetoed, "seed {seed}");
        assert_eq!(bed.submit_at(frame(id, seed), OTHER, Duration::ZERO), None);
        theirs.push(picked(&bed.answer().await));
    }
    // The veto is the asking gateway's alone.
    assert!(theirs.contains(&vetoed), "{theirs:?}");
}

/// The kinds of a gateway's vetoes reach its decisions. A pool it doubts
/// for liveness on every worker is steered: a doubt is not an outage. A
/// breaker's open window on each worker refuses the pool, as the gateway
/// refuses it. One doubted worker among held ones takes every request,
/// a kind this build has no name for holds as a breaker does, and
/// another gateway reads none of it.
#[tokio::test]
async fn a_pool_behind_the_gateways_open_breakers_is_refused_and_one_it_doubts_is_steered() {
    const BREAKER: i32 = proto::gateway_veto::Kind::Breaker as i32;
    const LIVENESS: i32 = proto::gateway_veto::Kind::Liveness as i32;
    let mut bed = leading([]);
    let now = bed.script.taken;
    let until = now + Duration::from_secs(60);
    let workers = bed.workers.clone();
    let fence = bed.elector.fence();
    let gateways = Arc::new(Gateways::new(GRACE, 1_024, fence, Arc::clone(&bed.store)));
    let seat = gateways
        .welcome(GATEWAY, "gateway-a", 1, now)
        .expect("the table takes the Hello");
    let veto = |worker: WorkerId, kind: i32, until: Option<Instant>| {
        assert_eq!(seat.veto(worker, kind, until, now), Vetoed::Shown);
    };
    let shed_at = |answer: &proto::DecideAnswer| match outcome(answer) {
        Outcome::Shed(shed) => (shed.reason(), shed.emptied_by.clone()),
        other => panic!("a shed: {other:?}"),
    };
    let at_health = (proto::ShedReason::NoEndpoints, "healthy".to_owned());
    for &worker in &workers {
        veto(worker, LIVENESS, Some(until));
    }
    let steered = bed.ask(frame(1, 1)).await;
    assert!(workers.contains(&picked(&steered)));
    for &worker in &workers {
        veto(worker, BREAKER, Some(until));
    }
    assert_eq!(shed_at(&bed.ask(frame(2, 2)).await), at_health);
    // One breaker closes: its worker is doubted alone.
    let doubted = workers[1];
    veto(doubted, BREAKER, None);
    for (id, seed) in (3..=10).zip(3..) {
        let answer = bed.ask(frame(id, seed)).await;
        assert_eq!(picked(&answer), doubted, "seed {seed}");
    }
    veto(doubted, 7, Some(until));
    assert_eq!(shed_at(&bed.ask(frame(11, 11)).await), at_health);
    assert_eq!(bed.submit_at(frame(12, 12), OTHER, Duration::ZERO), None);
    assert!(workers.contains(&picked(&bed.answer().await)));
}

/// The pair plan's document of the cases, as generation 1: a request of
/// the prefill or the decode phase under one pair binding, each side by
/// `pick`, and a single request under the same profile.
fn pair_plan(catalog: &StageRegistry, pick: &str) -> AppliedSpec {
    let document = format!(
        r#"{{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": {{ "p": {{ "pick": "{pick}" }} }},
              "bindings": [
                {{ "match": {{ "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
                  "profiles": {{ "prefill": "p", "decode": "p" }} }},
                {{ "profile": "p" }}
              ]
            }}"#
    );
    let document: PolicySpec = serde_json::from_str(&document).expect("a document of this build");
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, catalog, &[])
        .expect("the catalog holds the stage")
}

/// A controller decides a single request, a pair at its prefill and a
/// decode call. Under a pair plan a request of the decode phase that
/// names no lease, and one of the encode phase that does not state
/// `DecideRequest.encode`, have no decision: `Busy` with no reason,
/// counted by its cause, never one side's decision and never a shed,
/// which is a policy's word. Its gateway decides it. A prompt carried
/// as text is never decided as a request with no prefix.
#[tokio::test]
async fn a_request_of_a_decode_or_an_encode_phase_and_a_prompt_as_text_have_no_decision() {
    let spec = pair_plan(&StageRegistry::builtin(), "random");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let sides = [proto::Phase::Decode, proto::Phase::Encode];
    for (id, phase) in (1..).zip(sides) {
        let answer = bed.ask(phased(frame(id, 77), phase)).await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice(), answer.term),
            (Reason::Unspecified, &[id][..], 1),
            "{phase:?}"
        );
    }
    // Text the controller is to hash: never decided as a request with
    // no prefix.
    let mut text = frame(4, 77);
    text.facts
        .as_mut()
        .expect("the case's frame has facts")
        .prompt_text = b"a prompt".to_vec();
    let answer = bed.ask(text).await;
    assert_eq!(
        (busy(&answer), answer.request_id.as_slice(), answer.term),
        (Reason::Unspecified, &[4][..], 1)
    );
    let counted = |count: &AtomicU64| count.load(Ordering::Relaxed);
    let DecideStats {
        busy: busies,
        faults,
        pair_sides,
        texts,
        ..
    } = &*bed.counts;
    assert_eq!(
        (
            counted(faults),
            counted(pair_sides),
            counted(texts),
            bed.counts.hashes_cut()
        ),
        (0, 2, 1, 0)
    );
    // Each of the three is a `Busy` with no reason, and counted as one.
    assert_eq!(busies.each_ref().map(counted), [3, 0, 0, 0, 0, 0, 0, 0, 0]);
    // The same document's single request is decided, and so is its
    // pair at the prefill.
    for phase in [proto::Phase::Single, proto::Phase::Prefill] {
        let answer = bed.ask(phased(frame(5, 77), phase)).await;
        assert!(matches!(outcome(&answer), Outcome::Pick(_)), "{answer:?}");
    }
    assert_eq!(counted(&bed.counts.pair_sides), 2);
}

/// The bed's one worker of `role`: its fleet holds one a side.
fn side_of(bed: &Bed, role: Role) -> WorkerId {
    let (_, rows) = bed.store.all_rows();
    let row = rows.iter().find(|row| row.facts.role == role);
    row.expect("the fleet holds one worker a side").id
}

/// The leases a decision's answer carries, as the wire states each: its
/// id, its side and its session key.
fn leases_of(answer: &proto::DecideAnswer) -> Vec<(u64, proto::Side, u64)> {
    match &answer.answer {
        Some(Answer::Decision(decision)) => {
            let leases = decision.leases.iter();
            let stated = leases.map(|lease| (lease.id, lease.side(), lease.session_key));
            stated.collect()
        }
        other => panic!("a decision: {other:?}"),
    }
}

/// A pair at its prefill is decided once, for both sides, over its
/// model's prefills and decodes, and answered a lease a side under one
/// id, the prefill side's first, each under the session key of a
/// request of its side's phase: one record on file with its decode
/// side issued. A frame that states the later decode is answered the
/// prefill side's lease alone, and its decode side stands chosen.
#[tokio::test]
async fn a_pair_at_its_prefill_is_answered_a_lease_a_side_under_one_id() {
    let spec = pair_plan(&StageRegistry::builtin(), "random");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let (prefill, decode) = (side_of(&bed, Role::Prefill), side_of(&bed, Role::Decode));
    let id = |count: u64| 1 << 32 | count;
    let on_file = |bed: &Bed, lease: u64| {
        let store: &dyn LeasesStore = bed.store.as_ref();
        let sides = store.sides(GATEWAY, LeaseId(lease)).expect("on file");
        (sides.record.decode, sides.decode.map(|side| side.state))
    };
    // The case's frame at `phase` under one plain key, and the session
    // key the frame's own reader states of it for that phase.
    let keyed = |phase: proto::Phase| {
        let mut keyed = phased(frame(1, 1), phase);
        let facts = keyed.facts.as_mut().expect("the case's frame has facts");
        facts.session_key = Some(proto::Identity {
            form: Some(proto::identity::Form::Plain("conversation-7".to_owned())),
        });
        keyed
    };
    let key_at = |phase: proto::Phase| {
        let request = checked(keyed(phase));
        let mut lists = wire::Lists::default();
        let read = lists.of(&request);
        request.facts(&read, 0, 0).session().get()
    };

    let answer = bed.ask(keyed(proto::Phase::Prefill)).await;
    let Outcome::Pick(pick) = outcome(&answer) else {
        panic!("a pick: {answer:?}");
    };
    let first = pick.targets.first().expect("a target");
    let pair = first
        .pair
        .as_ref()
        .expect("a pair's target names its decode");
    assert_eq!(
        (WorkerId(first.id), WorkerId(pair.decode_id)),
        (prefill, decode)
    );
    let keys = [proto::Phase::Prefill, proto::Phase::Decode].map(key_at);
    assert!(
        keys[0] != 0 && keys[1] != 0 && keys[0] != keys[1],
        "{keys:?}"
    );
    let sides = [
        (id(1), proto::Side::Prefill, keys[0]),
        (id(1), proto::Side::Decode, keys[1]),
    ];
    assert_eq!(leases_of(&answer), sides, "the prefill side's first");
    assert_eq!(
        on_file(&bed, id(1)),
        (Some(DecodeMark), Some(LeaseState::ISSUED))
    );
    assert_eq!(bed.store.live(GATEWAY), 1, "one record for both sides");

    let mut later = phased(frame(2, 2), proto::Phase::Prefill);
    later.decode_later = true;
    let answer = bed.ask(later).await;
    assert_eq!(leases_of(&answer), [(id(2), proto::Side::Prefill, 0)]);
    assert_eq!(
        on_file(&bed, id(2)),
        (Some(DecodeMark), Some(LeaseState::Chosen))
    );
}

/// A pair's frame is counted as a single's: the workers its gateway's
/// wires take, on both sides, and a frame with more block hashes than
/// its document reads. The fleet's two sides state no wire, so a
/// gateway that states one takes both workers, and the frame is shed
/// under the wires' stage.
#[tokio::test]
async fn a_pairs_frame_counts_the_workers_its_wires_take_on_both_sides_and_its_cut_hashes() {
    let spec = pair_plan(&StageRegistry::builtin(), "random");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let mut wired = constrained(phased(frame(1, 1), proto::Phase::Prefill), 0b010, None);
    let facts = wired.facts.as_mut().expect("the case's frame has facts");
    facts.block_hashes = vec![1, 2];
    let answer = bed.ask(wired).await;
    let shed = declined(proto::ShedReason::NoEndpoints, "wires");
    assert_eq!(outcome(&answer), &shed);
    assert_eq!((bed.counts.undialed(), bed.counts.hashes_cut()), (2, 1));
}

/// A prefill frame, and a decode frame that names no lease, stand alone
/// under a plan that pairs no side. Each is decided over the list of its
/// own phase's role, here the fleet's one worker of that role and none
/// of its three single workers. Its pick is filed a single's lease: one
/// lease of the single side on the wire, one record with no decode side
/// on file, and its worker booked. Neither frame is counted with the
/// sides. An encode frame that does not state `DecideRequest.encode`
/// has no decision under this plan either: `Busy` with no reason,
/// counted, and nothing filed. A model that has no worker of the
/// frame's role is shed with no endpoint, as a single request of a
/// model nobody serves is.
#[tokio::test]
async fn a_prefill_or_a_decode_frame_that_stands_alone_is_decided_over_its_own_phases_list() {
    let mut bed = leading([]);
    let id = |count: u64| 1 << 32 | count;
    let sides = [
        (proto::Phase::Prefill, Role::Prefill),
        (proto::Phase::Decode, Role::Decode),
    ];
    for (n, (phase, role)) in (1..).zip(sides) {
        let worker = side_of(&bed, role);
        let answer = bed.ask(phased(frame(n, 1), phase)).await;
        assert_eq!(picked(&answer), worker, "{phase:?}");
        let lease = id(u64::from(n));
        assert_eq!(sides_of(&answer), [(lease, proto::Side::Single, 0)]);
        let store: &dyn LeasesStore = bed.store.as_ref();
        let filed = store.sides(GATEWAY, LeaseId(lease)).expect("on file");
        let no_decode = filed.record.decode.is_none() && filed.decode.is_none();
        assert_eq!(
            (filed.record.target, no_decode),
            (worker, true),
            "{phase:?}: a single's record"
        );
        assert_eq!(booked(&bed, [worker]), [1], "{phase:?}");
    }
    assert_eq!(bed.store.live(GATEWAY), 2);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
    let encode = phased(frame(3, 1), proto::Phase::Encode);
    assert_eq!(busy(&bed.ask(encode).await), Reason::Unspecified);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 1);
    // A model served by a single worker alone: no prefill and no decode
    // of it stands in the registry's lists.
    serving(
        &bed.store,
        &bed.reads.objects,
        "s9",
        &["model-of-singles"],
        Role::Single,
        WireModes::NONE,
    );
    let none = declined(proto::ShedReason::NoEndpoints, "none");
    for (id, phase) in (9..).zip([proto::Phase::Prefill, proto::Phase::Decode]) {
        let lone = phased(frame_for("model-of-singles", id, 9), phase);
        assert_eq!(*outcome(&bed.ask(lone).await), none, "{phase:?}");
    }
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.live(GATEWAY), 2);
}

/// The decode phase's own lookup says whether a decode frame that names
/// no lease stands alone. Under the pair plan it is a pair's side with
/// no lease to decide behind: `Busy` with no reason, counted with the
/// sides. With a binding of one profile listed ahead of the pair's for
/// the decode phase, the same frame is decided alone over the decodes
/// and filed a single's lease, and the model's prefill frame is still
/// its pair's one request.
#[tokio::test]
async fn a_binding_of_one_profile_ahead_of_the_pairs_decides_a_decode_frame_alone() {
    let ahead = |alone: &str| {
        let document = format!(
            r#"{{
                  "apiVersion": "smg.policy/v1",
                  "generation": 1,
                  "profiles": {{ "p": {{ "pick": "random" }} }},
                  "bindings": [
                    {alone}
                    {{ "match": {{ "phase": ["prefill", "decode"] }}, "plan": "pd-pair",
                      "profiles": {{ "prefill": "p", "decode": "p" }} }},
                    {{ "profile": "p" }}
                  ]
                }}"#
        );
        let document: PolicySpec =
            serde_json::from_str(&document).expect("a document of this build");
        applied(&document)
    };
    let decode = phased(frame(1, 1), proto::Phase::Decode);
    let mut paired = bed(Some(ahead("")), true, (2, 8), []);
    assert_eq!(busy(&paired.ask(decode.clone()).await), Reason::Unspecified);
    assert_eq!(paired.counts.pair_sides.load(Ordering::Relaxed), 1);

    let alone = r#"{ "match": { "phase": ["decode"] }, "profile": "p" },"#;
    let mut bed = bed(Some(ahead(alone)), true, (2, 8), []);
    let (prefill, decoder) = (side_of(&bed, Role::Prefill), side_of(&bed, Role::Decode));
    let answer = bed.ask(decode).await;
    assert_eq!(picked(&answer), decoder);
    let id = |count: u64| 1 << 32 | count;
    assert_eq!(sides_of(&answer), [(id(1), proto::Side::Single, 0)]);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
    // The prefill lookup resolves to the pair still: both sides run
    // under the pair binding, whatever binding the decode phase names
    // first.
    let answer = bed.ask(phased(frame(2, 2), proto::Phase::Prefill)).await;
    let pair = [
        (id(2), proto::Side::Prefill, 0),
        (id(2), proto::Side::Decode, 0),
    ];
    assert_eq!(sides_of(&answer), pair);
    assert_eq!(booked(&bed, [prefill, decoder]), [1, 2]);
}

/// A prefill frame, and a decode frame that names no lease, of a model
/// that no binding matches stand alone too: the document pairs the
/// sides of another model, and no plan pairs a side of this one. The
/// model has workers of each role here, and each frame is declined as
/// the model's single request is: by the policy, at the stage
/// `binding`. Nothing is filed, and neither frame is counted with the
/// sides.
#[tokio::test]
async fn a_prefill_or_a_decode_frame_of_a_model_no_binding_matches_is_declined_as_its_single_is() {
    let document = r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 1,
          "profiles": { "p": { "pick": "random" } },
          "bindings": [
            { "match": { "model": "model-of-another", "phase": ["prefill", "decode"] },
              "plan": "pd-pair", "profiles": { "prefill": "p", "decode": "p" } },
            { "match": { "model": "model-of-another" }, "profile": "p" }
          ]
        }"#;
    let document: PolicySpec = serde_json::from_str(document).expect("a document of this build");
    let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
    let unbound = declined(proto::ShedReason::PolicyDeclined, "binding");
    let phases = [
        proto::Phase::Single,
        proto::Phase::Prefill,
        proto::Phase::Decode,
    ];
    for (n, phase) in (1..).zip(phases) {
        let answer = bed.ask(phased(frame(n, 1), phase)).await;
        assert_eq!(*outcome(&answer), unbound, "{phase:?}");
    }
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
    assert_eq!(bed.store.live(GATEWAY), 0);
}

/// What the engine told a stage of a lease's sides: an issue and a
/// release, each by the side and the worker its lease books.
type SideCalls = Vec<(Side, WorkerId, Option<ReleaseReason>)>;

/// The `random` picker, keeping what the engine tells it of each side.
#[derive(Debug)]
struct Sided(Arc<Mutex<SideCalls>>);

impl Lifecycle for Sided {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push((lease.side, lease.booked, None));
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.0.lock().push((lease.side, lease.booked, Some(reason)));
    }
}

impl Picker for Sided {
    fn name(&self) -> &'static str {
        "sided"
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

/// A pair's answer that does not leave gives its lease up whole: the
/// record leaves the file with its decode side, and the engine that
/// heard each side's issue hears each side's release. A decode side
/// its frame asked for later was told no issue, and is told no
/// release. Here the decision's last read of the clock shows its
/// budget gone.
#[tokio::test]
async fn a_pairs_answer_that_does_not_leave_gives_up_both_sides_of_its_lease() {
    let late = Some(ReleaseReason::LateDecision);
    for later in [false, true] {
        let told = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&told);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("sided", move || Box::new(Sided(Arc::clone(&kept))))
            .expect("a free name");
        let steps = [Duration::ZERO, BUDGET].map(Step::At);
        let mut bed = bed(Some(pair_plan(&catalog, "sided")), true, (2, 8), steps);
        let (prefill, decode) = (side_of(&bed, Role::Prefill), side_of(&bed, Role::Decode));
        let mut ask = phased(frame(1, 1), proto::Phase::Prefill);
        ask.decode_later = later;
        let answer = bed.ask(ask).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
        assert_eq!(bed.store.live(GATEWAY), 0);
        let both = [
            (Side::Prefill, prefill, None),
            (Side::Decode, decode, None),
            (Side::Prefill, prefill, late),
            (Side::Decode, decode, late),
        ];
        let heard = both
            .into_iter()
            .filter(|(side, ..)| !later || *side == Side::Prefill);
        let heard: SideCalls = heard.collect();
        assert_eq!(*told.lock(), heard, "a later decode: {later}");
    }
}

/// A scorer that ranks every worker alike, keeping what the engine
/// tells it of each side of a lease.
#[derive(Debug)]
struct HeardSides(Arc<Mutex<SideCalls>>);

impl Lifecycle for HeardSides {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push((lease.side, lease.booked, None));
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.0.lock().push((lease.side, lease.booked, Some(reason)));
    }
}

impl Scorer for HeardSides {
    fn name(&self) -> &'static str {
        "heard-sides"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// The bed of the decode call's cases: a pair plan whose sides take the
/// worker with the fewest requests in flight, over the fleet's prefill
/// and its decode beside one more decode, behind one decider. With the
/// bed: what the engine told a stage of the decode side's profile, and
/// the prefill.
fn later_bed() -> (Bed, Arc<Mutex<SideCalls>>, WorkerId) {
    later_bed_of(1)
}

/// [`later_bed`] behind `deciders` deciders.
fn later_bed_of(deciders: usize) -> (Bed, Arc<Mutex<SideCalls>>, WorkerId) {
    let told = Arc::new(Mutex::new(Vec::new()));
    let bed = bed(Some(later_spec(1, &told)), true, (deciders, 8), []);
    ready(
        &bed.store,
        &bed.reads.objects,
        "d1",
        MODEL,
        Role::Decode,
        WireModes::NONE,
    );
    let prefill = side_of(&bed, Role::Prefill);
    (bed, told, prefill)
}

/// The document of [`later_bed`] as `generation`, whose decode side
/// keeps what its engine tells it in `told`.
fn later_spec(generation: u64, told: &Arc<Mutex<SideCalls>>) -> AppliedSpec {
    later_spec_with(generation, told, |_| {})
}

/// [`later_spec`] with `edit` made to its document ahead of its check.
fn later_spec_with(
    generation: u64,
    told: &Arc<Mutex<SideCalls>>,
    edit: impl FnOnce(&mut PolicySpec),
) -> AppliedSpec {
    later_spec_in(StageRegistry::builtin(), generation, told, edit)
}

/// [`later_spec_with`] over `catalog`: this build's stages and a case's
/// own beside them.
fn later_spec_in(
    mut catalog: StageRegistry,
    generation: u64,
    told: &Arc<Mutex<SideCalls>>,
    edit: impl FnOnce(&mut PolicySpec),
) -> AppliedSpec {
    hearing_sides(&mut catalog, told);
    later_spec_over(&catalog, generation, edit)
}

/// Adds to `catalog` the stage of [`later_bed`]'s decode side, which
/// keeps what its engine tells it in `told`.
fn hearing_sides(catalog: &mut StageRegistry, told: &Arc<Mutex<SideCalls>>) {
    let kept = Arc::clone(told);
    catalog
        .scorer("heard-sides", move || {
            Box::new(HeardSides(Arc::clone(&kept)))
        })
        .expect("a free name");
}

/// The document of [`later_bed`] as `generation`, validated over
/// `catalog`, which holds its decode side's stage ([`hearing_sides`]).
/// Two documents validated over one catalog are a controller's: the
/// later one keeps the stages of a profile both state alike.
fn later_spec_over(
    catalog: &StageRegistry,
    generation: u64,
    edit: impl FnOnce(&mut PolicySpec),
) -> AppliedSpec {
    let mut document: PolicySpec = serde_json::from_str(
        r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": {
                "side": { "pick": "count-first" },
                "told": { "score": ["heard-sides"], "pick": "count-first" }
              },
              "bindings": [
                { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
                  "profiles": { "prefill": "side", "decode": "told" } },
                { "profile": "side" }
              ]
            }"#,
    )
    .expect("a document of this build");
    document.generation = generation;
    edit(&mut document);
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, catalog, &[])
        .expect("the catalog holds the stage")
}

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

/// A pair asked with its decode later, and its prefill dispatched: the
/// lease's id and the first pick's two decodes, the chosen one first.
/// A second pair asked whole then books the chosen decode, so the
/// fewest in flight sends a call to the other: that pair's lease comes
/// last.
async fn held(bed: &mut Bed) -> (u64, [WorkerId; 2], u64) {
    let answer = bed.ask(pair_frame(1, true)).await;
    let (targets, sides) = pair_of(&answer);
    let id = sides.first().expect("the prefill's lease").0;
    let decodes = [0, 1].map(|at| WorkerId(targets[at].1.expect("a pair's target")));
    // One seed, the same counts on the decodes: the whole pair takes
    // the same.
    let whole = bed.ask(pair_frame(2, false)).await;
    let (targets, sides) = pair_of(&whole);
    assert_eq!(targets[0].1, Some(decodes[0].0));
    let whole = sides.first().expect("the whole pair's lease").0;
    let at = bed.script.taken;
    let prefill = sent(Side::Prefill);
    assert!(bed.store.dispatched(GATEWAY, LeaseId(id), prefill, at));
    (id, decodes, whole)
}

/// The decode side of `lease` as the file holds it.
fn decode_on_file(bed: &Bed, lease: u64) -> (LeaseState, WorkerId) {
    let sides = bed.store.sides(GATEWAY, LeaseId(lease));
    let decode = sides.and_then(|sides| sides.decode);
    let decode = decode.expect("the decode side on file");
    (decode.state, decode.target)
}

/// Files one issued lease of [`GATEWAY`] on `target`, under `id`.
fn file(store: &MemoryStore, id: u64, target: WorkerId, now: Instant) {
    let leases: &dyn LeasesStore = store;
    let filed = leases.issue(crate::store::LeaseRecord {
        lease: Lease {
            id: LeaseId(1 << 63 | id),
            ..Lease::default()
        },
        gateway: GATEWAY,
        request: id,
        target,
        picked: crate::store::PickRecord::default(),
        decode: None,
        untold: false,
        term: 1,
        issued_at: now,
        dispatch_deadline: now + DISPATCH_TTL,
        state: LeaseState::ISSUED,
    });
    assert_eq!(filed, Ok(()));
}

/// The decode call answers a decode pick under the pair's own lease:
/// decided on the counts of its instant, its targets name the held
/// prefill, and the new pick's decode is booked in the first's place.
#[tokio::test]
async fn a_served_decode_call_answers_a_decode_pick_under_the_same_lease() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let heard = told.lock().len();
    let answer = bed.ask(call(1, id)).await;
    let (targets, sides) = pair_of(&answer);
    assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
    // The chosen decode took a lease since: the call picks the other.
    assert_eq!(targets[0], (prefill.0, Some(other.0)));
    assert!(targets.iter().all(|target| target.0 == prefill.0));
    assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, other));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 1]);
    assert_eq!(told.lock()[heard..], [(Side::Decode, other, None)]);
    assert_eq!(asked_again(&bed), [0, 0]);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
}

/// A decode call's answer says no word of a term's mark. Its lease's
/// id is its pair's, minted under the term before and at that term's
/// mark: the term that answers the call minted no id yet, and a word
/// there would hand a fresh term over.
#[tokio::test]
async fn a_decode_call_on_a_lease_at_an_earlier_terms_mark_says_no_word() {
    let (mut bed, _, _) = later_bed();
    let (words, go) = heard(&bed);
    for _ in 0..8 {
        go.send(()).expect("the decider holds the other end");
    }
    // The pair's lease takes the id at its term's mark, and its answer
    // says that term's word.
    bed.store.spend_ids(u32::MAX - TERM_MARK + 1);
    let (id, _, _) = held(&mut bed).await;
    assert_eq!(id, 1 << 32 | u64::from(TERM_MARK));
    words.recv_timeout(PATIENCE).expect("the first term's word");
    // The bed's candidate, alone over its lease, leaves and takes again
    // at one instant, as its election would. The pair's lease stays on
    // its file under the new term.
    let taken = bed.script.taken;
    let left = bed.elector.step_down(None, taken);
    assert!(left.expect("a planned hand-over"));
    let took = bed.elector.tick(taken).expect("the take");
    assert_eq!(took, TickOutcome::Acquired { term: 2 });
    bed.store.set_term(2);
    let answer = bed.ask(call(3, id)).await;
    assert_eq!(answer.term, 2);
    assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
    // The answer after the call's shows its decider said what it had
    // to say: nothing. That answer's own id is the new term's first.
    let next = bed.ask(pair_frame(4, false)).await;
    assert_eq!(sides_of(&next)[0].0, 2 << 32 | 1);
    assert!(words.try_recv().is_err(), "a call mints no id");
}

/// A decode call's frame is counted as a pair's: the decodes its
/// gateway's wires take, and a frame with more block hashes than its
/// document reads. The fleet's decodes state no wire, so a gateway
/// that states one takes both, and the call is shed under the wires'
/// stage with its side still chosen.
#[tokio::test]
async fn a_decode_calls_frame_counts_the_decodes_its_wires_take_and_its_cut_hashes() {
    let (mut bed, _, _) = later_bed();
    let (id, [chosen, _], _) = held(&mut bed).await;
    assert_eq!((bed.counts.undialed(), bed.counts.hashes_cut()), (0, 0));
    let mut wired = constrained(call(1, id), 0b010, None);
    let facts = wired.facts.as_mut().expect("the case's frame has facts");
    facts.block_hashes = vec![1, 2];
    let answer = bed.ask(wired).await;
    let shed = declined(proto::ShedReason::NoEndpoints, "wires");
    assert_eq!(outcome(&answer), &shed);
    assert_eq!((bed.counts.undialed(), bed.counts.hashes_cut()), (2, 1));
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
}

/// A decode call can pick a decode the store's world does not list yet,
/// as a single request's decision can pick such a worker: the worker
/// objects follow a registration as its write lands. The side's issue
/// books the worker's count all the same, by the slot the registration
/// filed.
#[tokio::test]
async fn a_decode_calls_pick_inside_its_workers_registration_window_books_its_count() {
    let (mut bed, _, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    // One more decode, written ready and never published: the build
    // that lists it is still owed. A lease filed on the other
    // decode leaves the new one alone with nothing in flight.
    let key = String::from("http://late:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role: Role::Decode,
        models: vec![MODEL.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    let late = rewritten(&bed, facts);
    let row = WorkersStore::get(&*bed.store, late).expect("the row");
    bed.reads
        .objects
        .connect_row(&bed.store, late, row.revision);
    let listed = bed.store.snapshot().booked_of(late);
    assert_eq!(listed, None, "no world lists the worker yet");
    file(&bed.store, 1, other, bed.script.taken);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(pair_of(&answer).0[0], (prefill.0, Some(late.0)));
    assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, late));
    assert_eq!(booked(&bed, [prefill, chosen, other, late]), [2, 1, 1, 1]);
}

/// What [`Ends`] ends: the prefill side of a lease of [`GATEWAY`]'s on
/// file in a store, once.
type Armed = Arc<Mutex<Option<(Weak<MemoryStore>, LeaseId)>>>;

/// A filter that keeps every worker and, as a decision opens it, ends
/// what its case armed it with: what another hand does to a lease
/// between a decode call's first test of it and the call's issue.
#[derive(Debug)]
struct Ends(Armed);

impl Lifecycle for Ends {}

impl DecisionScope for Ends {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        let Some((store, lease)) = self.0.lock().take() else {
            return;
        };
        let store = store.upgrade().expect("the case's store");
        let (prefill, gone) = (Some(Side::Prefill), ReleaseReason::ClientGone);
        assert!(store.released(GATEWAY, lease, prefill, gone));
    }

    fn close(&self) {}
}

impl Filter for Ends {
    fn name(&self) -> &'static str {
        "ends"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self)
    }
}

/// A lease that moves between a decode call's first test and its issue
/// refuses the issue, and the call leaves as the `Busy` of that
/// refusal: here the lease's prefill ends while the call's stages run,
/// and takes the lease off the file with its chosen side. Nothing is
/// issued, no stage hears an issue, and no count is booked.
#[tokio::test]
async fn a_lease_that_ends_under_a_decode_calls_decision_refuses_its_issue_by_name() {
    let armed = Armed::default();
    let cell = Arc::clone(&armed);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("ends", move || Box::new(Ends(Arc::clone(&cell))))
        .expect("a free name");
    let told = Arc::new(Mutex::new(Vec::new()));
    let spec = later_spec_in(catalog, 1, &told, |document| {
        let profile = document.profiles.get_mut("told");
        let profile = profile.expect("the decode side's profile");
        profile.filter.push("ends".into());
    });
    let mut bed = bed(Some(spec), true, (1, 8), []);
    ready(
        &bed.store,
        &bed.reads.objects,
        "d1",
        MODEL,
        Role::Decode,
        WireModes::NONE,
    );
    let prefill = side_of(&bed, Role::Prefill);
    let (id, [chosen, other], _) = held(&mut bed).await;
    *armed.lock() = Some((Arc::downgrade(&bed.store), LeaseId(id)));
    let heard = told.lock().len();
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::DecodeNoLease);
    assert!(armed.lock().is_none(), "the call's stages ran");
    assert_eq!(bed.store.sides(GATEWAY, LeaseId(id)), None);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [1, 1, 0]);
    assert_eq!(told.lock().len(), heard, "no stage heard an issue");
}

/// A decode frame that names no lease has no decision under a plan that
/// pairs the decode phase: `Busy` with no reason, counted with the
/// sides, and nothing of a pair's lease on file moves.
#[tokio::test]
async fn a_decode_frame_that_names_no_lease_has_no_decision_and_is_counted() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let (filed, heard) = (bed.store.sides(GATEWAY, LeaseId(id)), told.lock().len());
    let alone = phased(frame(1, 77), proto::Phase::Decode);
    assert_eq!(busy(&bed.ask(alone).await), Reason::Unspecified);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.sides(GATEWAY, LeaseId(id)), filed);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    assert_eq!(told.lock().len(), heard);
}

/// A served decode call's kept record names the pair binding the call
/// decided under, whatever phase its frame states. A binding of the
/// decode phase stands ahead of the pair's here: it decides a decode
/// asked alone, and no pair's.
#[tokio::test]
async fn a_served_decode_calls_record_names_its_pairs_binding() {
    use crate::{
        document::spec::{BindingSpec, Match},
        engine::view::Phase,
    };

    let (mut bed, told, _) = later_bed();
    let sampled = later_spec_with(2, &told, |document| {
        let ahead = BindingSpec {
            matcher: Match {
                phase: vec![Phase::Decode],
                ..Match::default()
            },
            profile: Some("side".into()),
            ..BindingSpec::default()
        };
        document.bindings.insert(0, ahead);
        document.explain.sample = 1;
    });
    bed.store.apply(sampled).expect("a later generation");
    let (id, _, _) = held(&mut bed).await;
    // The call is asked under an id of its own: the pair's two first
    // frames left their records under the same binding, and a record
    // looked up by either's id would be that frame's.
    let no_record = bed.deciders.explain(1, GATEWAY, &[3]);
    assert!(no_record.is_none(), "no frame was asked under the id yet");
    let answer = bed.ask(phased(call(3, id), proto::Phase::Decode)).await;
    assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
    let record = bed.deciders.explain(answer.term, GATEWAY, &[3]);
    // The pair's binding is the document's second.
    assert_eq!(record.expect("the sampled call").binding, Some(1));
}

/// Each refusal of the call leaves as a `Busy` of its own reason,
/// never as a decision, and moves nothing: no lease under the id, a
/// pair filed whole, a prefill not yet sent, and a side a call issued.
#[tokio::test]
async fn each_refusal_of_the_call_is_a_busy_of_its_own_reason_and_moves_nothing() {
    let (mut bed, told, prefill) = later_bed();
    let lease = |answer: &proto::DecideAnswer| sides_of(answer)[0].0;
    let later = lease(&bed.ask(pair_frame(1, true)).await);
    let whole = lease(&bed.ask(pair_frame(2, false)).await);
    let store = Arc::clone(&bed.store);
    let decodes = ["d0", "d1"].map(|name| {
        let row = WorkersStore::by_key(&*store, &format!("http://{name}:8000"));
        row.expect("a decode of the fleet").id
    });
    let file = |bed: &Bed| {
        let sides = [later, whole].map(|id| store.sides(GATEWAY, LeaseId(id)));
        let counts = booked(bed, [prefill, decodes[0], decodes[1]]);
        (sides, counts, told.lock().len())
    };
    let rows = [
        (later + 100, Reason::DecodeNoLease),
        (whole, Reason::DecodeNotChosen),
        (later, Reason::DecodePrefillNotHeld),
    ];
    for (id, reason) in rows {
        let before = file(&bed);
        assert_eq!(busy(&bed.ask(call(1, id)).await), reason);
        assert_eq!(file(&bed), before, "{reason:?}");
    }
    // The fourth: a side a first call issued stands.
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, LeaseId(later), sent(Side::Prefill), at));
    let issued = bed.ask(call(1, later)).await;
    assert!(matches!(outcome(&issued), Outcome::Pick(_)));
    let before = file(&bed);
    assert_eq!(busy(&bed.ask(call(1, later)).await), Reason::DecodeStands);
    assert_eq!(file(&bed), before);
    // Each is counted at its own reason, once.
    let counted = |reason: Reason| bed.counts.busy[reason as usize].load(Ordering::Relaxed);
    let reasons = [
        Reason::DecodeNoLease,
        Reason::DecodeNotChosen,
        Reason::DecodePrefillNotHeld,
        Reason::DecodeStands,
    ];
    assert_eq!(reasons.map(counted), [1; 4]);
}

/// A call that sheds is a decision, as any shed is: nothing is issued
/// and the side stays chosen with the first pick's targets. A dispatch
/// then books the first pick's decode.
#[tokio::test]
async fn a_shed_at_the_call_is_a_decision_and_leaves_the_side_chosen() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let before = store.sides(GATEWAY, LeaseId(id));
    let heard = told.lock().len();
    let answer = bed.ask(constrained(call(1, id), 0, Some(0))).await;
    let Outcome::Shed(shed) = outcome(&answer) else {
        panic!("a shed: {answer:?}");
    };
    assert_eq!(shed.emptied_by, "subset-hint");
    assert!(sides_of(&answer).is_empty());
    assert_eq!(store.sides(GATEWAY, LeaseId(id)), before);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    assert_eq!(told.lock().len(), heard);
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 0]);
}

/// A decode issued under a term that ended returns to chosen. The
/// answer leaves as `Busy`, the lease stands as the call found it with
/// the first pick's targets, the new pick's booking is freed and the
/// engine hears that issue's release once. A dispatch then books the
/// first pick's decode.
#[tokio::test]
async fn a_decode_issued_under_a_term_that_ended_returns_to_chosen() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let before = store.sides(GATEWAY, LeaseId(id));
    let heard = told.lock().len();
    // The call's first read finds the fence open; its read at the
    // hand-over is past the renew deadline.
    let steps = [Step::At(Duration::ZERO), Step::At(FENCED)];
    bed.script.steps.lock().extend(steps);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::NotLeader);
    assert_eq!(store.sides(GATEWAY, LeaseId(id)), before);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let abandoned = Some(ReleaseReason::Abandoned);
    let issue = [
        (Side::Decode, other, None),
        (Side::Decode, other, abandoned),
    ];
    assert_eq!(told.lock()[heard..], issue);
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 0]);
}

/// A decode call whose decision's last read of the clock shows its
/// budget gone gives its issue back under the late reason, and is
/// counted as a decision that outlasted its budget.
#[tokio::test]
async fn a_decode_call_that_outlasts_its_budget_gives_its_issue_back() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let before = bed.store.sides(GATEWAY, LeaseId(id));
    let heard = told.lock().len();
    let steps = [Step::At(Duration::ZERO), Step::At(BUDGET)];
    bed.script.steps.lock().extend(steps);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.sides(GATEWAY, LeaseId(id)), before);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let late = Some(ReleaseReason::LateDecision);
    let issue = [(Side::Decode, other, None), (Side::Decode, other, late)];
    assert_eq!(told.lock()[heard..], issue);
}

/// A repeat of the call frees the lost issue and issues the side anew
/// under the same lease id. The word takes back the first call's
/// issue, ahead of the repeat's own decision, and the engine hears
/// that issue's release as abandoned.
#[tokio::test]
async fn a_repeat_of_the_call_frees_the_lost_issue_and_issues_the_side_anew() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let first = bed.ask(call(1, id)).await;
    assert_eq!(pair_of(&first).0[0].1, Some(other.0));
    // The loads move while the answer is lost: two more requests on
    // the decode the first call issued.
    for extra in [1, 2] {
        file(&store, extra, other, bed.script.taken);
    }
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 3]);
    let heard = told.lock().len();
    let repeat = bed.ask(again(call(1, id))).await;
    let (targets, sides) = pair_of(&repeat);
    assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
    assert_eq!(targets[0], (prefill.0, Some(chosen.0)));
    assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, chosen));
    // The first call's decode gave its booking back, and the repeat's
    // took one.
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 2]);
    let abandoned = Some(ReleaseReason::Abandoned);
    let words = [
        (Side::Decode, other, abandoned),
        (Side::Decode, chosen, None),
    ];
    assert_eq!(told.lock()[heard..], words);
    assert_eq!(asked_again(&bed), [0, 1]);
}

/// A repeat that sheds leaves the side chosen, as its first decision
/// filed it: the word took the lost issue back ahead of the repeat's
/// decision, whatever its answer. The gateway then sends the decode its
/// first pick named, and that dispatch books the side.
#[tokio::test]
async fn a_repeat_that_sheds_leaves_the_side_chosen_for_the_first_picks_dispatch() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let lease = LeaseId(id);
    let filed = store.sides(GATEWAY, lease);
    bed.ask(call(1, id)).await;
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 1]);
    assert_ne!(store.sides(GATEWAY, lease), filed);
    let repeat = again(constrained(call(1, id), 0, Some(0)));
    let answer = bed.ask(repeat).await;
    assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
    // The lease stands as its first decision filed it, both its parts:
    // the first pick's targets and the lease's dispatch deadline.
    assert_eq!(store.sides(GATEWAY, lease), filed);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let abandoned = Some(ReleaseReason::Abandoned);
    assert_eq!(told.lock().last(), Some(&(Side::Decode, other, abandoned)));
    assert_eq!(asked_again(&bed), [0, 1]);
    // The first pick's decode goes out: its dispatch books the side.
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, lease, sent(Side::Decode), at));
    let sent_to = (LeaseState::Dispatched { at }, chosen);
    assert_eq!(decode_on_file(&bed, id), sent_to);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 0]);
}

/// A repeat on a side still chosen is a first call: no call issued the
/// side, and the word ends nothing.
#[tokio::test]
async fn a_repeat_on_a_chosen_side_is_a_first_call() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let heard = told.lock().len();
    let answer = bed.ask(again(call(1, id))).await;
    let (targets, sides) = pair_of(&answer);
    assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
    assert_eq!(targets[0].1, Some(other.0));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 1]);
    assert_eq!(told.lock()[heard..], [(Side::Decode, other, None)]);
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// A repeat on a side its gateway named a dispatch for is refused as
/// standing: the word ends no side that was sent.
#[tokio::test]
async fn a_repeat_on_a_dispatched_side_is_refused_as_standing() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    bed.ask(call(1, id)).await;
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
    let file = || (store.sides(GATEWAY, LeaseId(id)), told.lock().len());
    let before = file();
    let answer = bed.ask(again(call(1, id))).await;
    assert_eq!(busy(&answer), Reason::DecodeStands);
    assert_eq!(file(), before);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 1]);
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// A decode frame runs no search by its id's fingerprint, with the
/// word or without: another request of the gateway under the same id
/// keeps the lease it waits under.
#[tokio::test]
async fn a_decode_frame_runs_no_search_by_its_ids_fingerprint() {
    let (mut bed, _, _) = later_bed();
    let (id, _, _) = held(&mut bed).await;
    // A single of the gateway under the id the call's frame carries:
    // its lease waits for a dispatch.
    let single = bed.ask(frame(9, 77)).await;
    let waits = LeaseId(lease_of(&single).id);
    let answer = bed.ask(again(call(9, id))).await;
    assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
    let leases: &dyn LeasesStore = bed.store.as_ref();
    assert!(leases.get(GATEWAY, waits).is_some(), "the single's lease");
    assert_eq!(asked_again(&bed), [1, 0]);
}

/// The issue that stands of `lease`'s decode side, taken back by the
/// case's own hand once the bed's decider left its work: a thread acts
/// at a file where nobody is at work, and beside nobody.
async fn taken_back_by_hand(store: &MemoryStore, lease: LeaseId) {
    let abandoned = ReleaseReason::Abandoned;
    settled(|| store.take_back_decode(GATEWAY, lease, abandoned)).await;
}

/// One decode call's order by the case's own hand, at the bed's first
/// instant: the issue its decider would undo.
fn issued_by_hand(bed: &Bed, frame: proto::DecideRequest) -> DecodeIssue {
    let request = checked(frame);
    let now = bed.script.taken;
    let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
    let (mut frames, part, mut caught) = (Default::default(), Part::default(), false);
    let lease = request.decode_of().expect("a decode call's frame");
    let Reads { store, objects, .. } = &bed.reads;
    let loads = Arc::new(LoadSnapshot::default());
    let decided = store.deciding(|world| {
        let theirs = (&mut frames, &loads, &part);
        decode_call(store, world, objects, &taken, lease, theirs, &mut caught)
    });
    match decided {
        Ok((_, Ok(Some(_)), FiledBeside::Issue(issue))) => issue,
        _ => panic!("a decode pick and its issue"),
    }
}

/// A decode call's answer that its stream does not take gives its one
/// issue back, and no more of the lease: the prefill that runs under
/// the same id keeps its booking. The side stands chosen with the
/// first pick's targets, the engine hears the issue's release as
/// abandoned, and the first pick's dispatch then books the side.
#[tokio::test]
async fn a_decode_calls_answer_its_stream_does_not_take_gives_its_issue_back() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let lease = LeaseId(id);
    let filed = store.sides(GATEWAY, lease);
    // A stream that ended takes no answer.
    let (sender, answers) = mpsc::channel(1);
    drop(answers);
    let job = Job {
        request: checked(call(1, id)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: sender,
    };
    assert!(bed.deciders.submit(job).is_none());
    // One decider decides a model's requests in order: it is past the
    // call, and past its answer's give-up, when it answers the next.
    bed.ask(frame(9, 9)).await;
    assert_eq!(store.sides(GATEWAY, lease), filed);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let abandoned = (Side::Decode, other, Some(ReleaseReason::Abandoned));
    assert!(told.lock().contains(&abandoned), "{:?}", told.lock());
    let at = bed.script.taken;
    assert!(store.dispatched(GATEWAY, lease, sent(Side::Decode), at));
    let sent_to = (LeaseState::Dispatched { at }, chosen);
    assert_eq!(decode_on_file(&bed, id), sent_to);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 2, 0]);
}

/// A decode side's answer of `lease` as its stream's queue holds it,
/// handed over with `beside` at `at`.
fn queued(lease: LeaseId, beside: wire::Beside, at: Instant) -> wire::Leased {
    wire::Leased {
        lease: Lease {
            id: lease,
            side: Side::Decode,
            ..Lease::default()
        },
        beside,
        left: DISPATCH_TTL,
        due: at + DISPATCH_TTL,
    }
}

/// A decode call's answer given up while a decider is at work at the
/// lease's file moves nothing and tells nothing on the thread that
/// gives it up: the word waits in the file's ring. The decider applies
/// it at its next take, ahead of its own decision: the issue goes back
/// and the engine hears its release there, so no stage hears a word of
/// a lease beside its decider at work.
#[tokio::test]
async fn an_answer_given_up_beside_a_decider_at_work_is_left_to_that_decider() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let lease = LeaseId(id);
    let filed = store.sides(GATEWAY, lease);
    let issue = issued_by_hand(&bed, call(1, id));
    let answer = queued(lease, wire::Beside::Issue(issue), bed.script.taken);
    let heard = told.lock().len();
    // The next decision is held at its first read of the clock: its
    // decider has the job at hand, at the lease's file.
    let (reached, held) = std_mpsc::sync_channel(1);
    let (go, word) = std_mpsc::sync_channel(1);
    let hold = Step::Hold { reached, go: word };
    bed.script.steps.lock().push_back(hold);
    assert_eq!(bed.submit(frame(9, 9)), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    let hand = Hand::new(GATEWAY);
    let late = ReleaseReason::LateDecision;
    // A take-back in place is refused beside the decider at work.
    assert!(!store.take_back_decode(GATEWAY, lease, late));
    answer.given_up(&store, &hand, late);
    assert_eq!(told.lock().len(), heard, "left to the decider at work");
    go.send(()).expect("the decider holds the other end");
    bed.answer().await;
    assert_eq!(store.sides(GATEWAY, lease), filed);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    let released = (Side::Decode, other, Some(late));
    assert_eq!(told.lock().get(heard), Some(&released));
    store.hand_back(hand);
}

/// A decode call is decided at its lease's file, whatever model it
/// names: its job goes to the decider that issued the lease, not to
/// the decider of the model the call names. A repeat under such a
/// name takes the call's issue back there, and the first pick's decode
/// stands chosen where the named model has no worker.
#[tokio::test]
async fn a_decode_call_is_decided_by_its_leases_decider_whatever_model_it_names() {
    let (mut bed, _, prefill) = later_bed_of(4);
    let (id, [chosen, other], _) = held(&mut bed).await;
    let home = format!("decide-{}", bed.deciders.of(MODEL));
    let elsewhere = (0..64)
        .map(|index| format!("model-{index}"))
        .find(|model| bed.deciders.of(model) != bed.deciders.of(MODEL))
        .expect("some model of sixty-four has another decider");
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, other));
    bed.script.readers.lock().clear();
    let mut repeat = again(call(1, id));
    let facts = repeat.facts.as_mut().expect("the case's frame has facts");
    facts.model = elsewhere;
    bed.ask(repeat).await;
    let readers = mem::take(&mut *bed.script.readers.lock());
    assert!(
        !readers.is_empty() && readers.iter().all(|reader| *reader == home),
        "{readers:?}"
    );
    assert_eq!(asked_again(&bed), [0, 1]);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
}

/// What an answer that does not leave gives up, by what its lease is
/// handed over with. A decode call's answer takes back its own issue
/// alone, by its count: behind a repeat's issue it moves nothing. Any
/// other answer's lease is released whole, both its sides.
#[tokio::test]
async fn an_answer_given_up_takes_back_its_own_issue_or_releases_its_lease_whole() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let lease = LeaseId(id);
    let filed = store.sides(GATEWAY, lease);
    let file = |bed: &Bed| {
        let counts = booked(bed, [prefill, chosen, other]);
        (store.sides(GATEWAY, lease), counts)
    };
    let at = bed.script.taken;
    let answer = |beside| queued(lease, beside, at);
    let hand = Hand::new(GATEWAY);
    let late = ReleaseReason::LateDecision;
    // The first call's answer, given up behind a repeat's issue.
    let first = answer(wire::Beside::Issue(issued_by_hand(&bed, call(1, id))));
    taken_back_by_hand(&store, lease).await;
    let repeat = answer(wire::Beside::Issue(issued_by_hand(&bed, call(1, id))));
    let issued = file(&bed);
    assert_eq!(issued.1, [2, 1, 1]);
    first.given_up(&store, &hand, late);
    assert_eq!(file(&bed), issued);
    // The repeat's answer, given up: its own issue goes back. Its word
    // is applied by the hand, or by a decider that leaves its work.
    repeat.given_up(&store, &hand, late);
    settled(|| file(&bed) == (filed, [2, 1, 0])).await;
    assert_eq!(told.lock().last(), Some(&(Side::Decode, other, Some(late))));
    // A pair's first answer, given up: the lease leaves whole.
    let whole = answer(wire::Beside::Decode(view::Session::NONE));
    whole.given_up(&store, &hand, late);
    store.hand_back(hand);
    assert_eq!(file(&bed), (None, [1, 1, 0]));
}

/// A give-up that names a chosen decode side ends it: a lost call that
/// runs late behind the give-up is refused as standing, with the word
/// or without, and books nobody.
#[tokio::test]
async fn a_give_up_that_names_a_chosen_side_ends_it() {
    let (mut bed, _, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let abandoned = ReleaseReason::Abandoned;
    assert!(store.released(GATEWAY, LeaseId(id), Some(Side::Decode), abandoned));
    let ended = (LeaseState::Released(abandoned), chosen);
    assert_eq!(decode_on_file(&bed, id), ended);
    for late in [call(1, id), again(call(1, id))] {
        assert_eq!(busy(&bed.ask(late).await), Reason::DecodeStands);
    }
    assert_eq!(decode_on_file(&bed, id), ended);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
}

/// A decode call whose model a reload bound to no pair plan is a
/// side's request still: it is not decided as a single. Its decision
/// sheds, nothing is issued and the side stays chosen.
#[tokio::test]
async fn a_decode_call_behind_a_reload_to_no_pair_plan_is_no_single() {
    let (mut bed, _, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let reloaded = store.apply(preset("random", 2));
    reloaded.expect("a later generation");
    let live = LeasesStore::live(&*store, GATEWAY);
    let answer = bed.ask(call(1, id)).await;
    assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
    assert!(sides_of(&answer).is_empty());
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(LeasesStore::live(&*store, GATEWAY), live);
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
}

/// A call's issue behind a reload is told to no engine, and neither is
/// its take-back. The engine of the call's instant heard no issue of
/// the lease, whose generation is the first decision's: it hears no
/// issue of the side and no release of it.
#[tokio::test]
async fn a_decode_issued_and_taken_back_behind_a_reload_is_told_to_no_engine() {
    let (mut bed, told, prefill) = later_bed();
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let before = told.lock().len();
    let reloaded = Arc::new(Mutex::new(Vec::new()));
    let applied = store.apply(later_spec(2, &reloaded));
    applied.expect("a later generation");
    let steps = [Step::At(Duration::ZERO), Step::At(FENCED)];
    bed.script.steps.lock().extend(steps);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::NotLeader);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    assert_eq!(*reloaded.lock(), []);
    assert_eq!(told.lock().len(), before);
}

/// A call's issue behind a reload that states the decode side's
/// profile as it stood is told to that profile's stage, which the
/// reload kept, and so is its take-back: the stage stood when the lease
/// was issued. Both documents are validated over one catalog, as a
/// controller validates its documents.
#[tokio::test]
async fn a_decode_issued_and_taken_back_behind_a_reload_is_told_to_the_stage_it_kept() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    hearing_sides(&mut catalog, &told);
    let first = later_spec_over(&catalog, 1, |_| {});
    let mut bed = bed(Some(first), true, (1, 8), []);
    ready(
        &bed.store,
        &bed.reads.objects,
        "d1",
        MODEL,
        Role::Decode,
        WireModes::NONE,
    );
    let prefill = side_of(&bed, Role::Prefill);
    let (id, [chosen, other], _) = held(&mut bed).await;
    let store = Arc::clone(&bed.store);
    let before = told.lock().len();
    let applied = store.apply(later_spec_over(&catalog, 2, |_| {}));
    applied.expect("a later generation");
    let steps = [Step::At(Duration::ZERO), Step::At(FENCED)];
    bed.script.steps.lock().extend(steps);
    let answer = bed.ask(call(1, id)).await;
    assert_eq!(busy(&answer), Reason::NotLeader);
    assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
    assert_eq!(booked(&bed, [prefill, chosen, other]), [2, 1, 0]);
    // The side's issue on the decode the call picked, then its release.
    let heard = told.lock()[before..].to_vec();
    let [(Side::Decode, issued, None), (Side::Decode, released, Some(_))] = heard[..] else {
        panic!("the stage kept heard {heard:?}");
    };
    assert_eq!((issued, released), (other, other));
}

/// A replica that does not lead says so for a request it has no
/// decision for, too.
#[tokio::test]
async fn a_replica_that_does_not_lead_says_so_before_it_reads_a_prompt_as_text() {
    let mut bed = bed(Some(preset("random", 1)), false, (2, 8), []);
    let mut text = frame(1, 1);
    text.facts
        .as_mut()
        .expect("the case's frame has facts")
        .prompt_text = b"a prompt".to_vec();
    assert_eq!(busy(&bed.ask(text).await), Reason::NotLeader);
    assert_eq!(bed.counts.texts.load(Ordering::Relaxed), 0);
}

/// A text is said before the budget is read: the count holds every
/// text a leader took, whatever was left of its budget.
#[tokio::test]
async fn a_prompt_as_text_whose_budget_is_gone_is_still_counted_as_a_text() {
    let mut bed = leading([Step::At(BUDGET)]);
    let mut text = frame(1, 1);
    text.facts
        .as_mut()
        .expect("the case's frame has facts")
        .prompt_text = b"a prompt".to_vec();
    assert_eq!(busy(&bed.ask(text).await), Reason::Unspecified);
    assert_eq!(bed.counts.texts.load(Ordering::Relaxed), 1);
    // The same instant with no text: the budget's own reason.
    let answer = bed.ask(frame(2, 2)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
}

/// The lines a case's own subscriber wrote.
#[derive(Clone, Default)]
struct Logged(Arc<Mutex<Vec<u8>>>);

impl io::Write for Logged {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A decider of the case's own, on the case's thread: a streak of
/// faults writes one line, and a decision that does not fault ends the
/// streak.
#[test]
fn a_streak_of_faults_writes_one_line_and_a_decision_ends_it() {
    // The second read of the clock panics: the view is built by then.
    let fault = || [Step::At(Duration::ZERO), Step::Panic];
    let steps = fault().into_iter().chain(fault()).chain(fault());
    let bed = bed(Some(preset("random", 1)), true, (1, 8), steps);
    let mut decider = deciding(&bed);
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    // A second subscriber stays registered beside it: with one alone a
    // line is on or off by the thread that reaches it first.
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let written = || {
        let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        text.matches("a decision panicked").count()
    };
    tracing::subscriber::with_default(subscriber, || {
        let ask = |decider: &mut Deciding, id| {
            let handed = decider.answer(checked(frame(id, 1)), GATEWAY, bed.script.taken);
            handed.frame()
        };
        for id in 1..=3 {
            assert_eq!(busy(&ask(&mut decider, id)), Reason::Unspecified);
        }
        assert_eq!(written(), 1, "one line for the streak");
        // A decision that does not fault ends the streak.
        let answer = ask(&mut decider, 4);
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        bed.script.steps.lock().extend(fault());
        assert_eq!(busy(&ask(&mut decider, 5)), Reason::Unspecified);
        assert_eq!(written(), 2, "a new streak, a new line");
    });
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 4);
}

/// A decider's thread holds the guard from its start: one that unwinds
/// is told, to a receiver taken after the word too; one that returns at
/// its queue's end, and a decision's own panic, are not.
#[tokio::test]
async fn a_thread_that_unwinds_is_told_and_one_that_returns_is_not() {
    let told = Arc::new(watch::Sender::new(false));
    let returned = watched(0, &told, || {}).expect("the thread starts").join();
    assert!(returned.is_ok() && !*told.subscribe().borrow());
    // No receiver stands at the unwind: the word is kept for the next.
    let unwound = watched(1, &told, || {
        panic!("the case's own panic, outside a decision")
    })
    .expect("the thread starts")
    .join();
    assert!(unwound.is_err() && *told.subscribe().borrow());

    // A decision that panics is caught: its decider's thread goes on,
    // and the deciders tell nothing.
    let spec = Some(preset("random", 1));
    let mut bed = bed(spec, true, (1, 8), [Step::Panic]);
    assert_eq!(busy(&bed.ask(frame(1, 1)).await), Reason::Unspecified);
    let answer = bed.ask(frame(2, 2)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert!(!*bed.deciders.ended().borrow());
    // The word a thread's unwind says is the one the deciders show, and
    // a receiver made after it reads it by its value.
    bed.deciders.tell_unwound();
    assert!(*bed.deciders.ended().borrow());
    let mut late = bed.deciders.ended();
    assert!(late.wait_for(|ended| *ended).await.is_ok());
}

#[tokio::test]
async fn a_decision_that_panics_is_answered_busy_with_no_reason_and_its_decider_decides_the_next() {
    let spec = Some(preset("random", 1));
    let mut bed = bed(spec, true, (1, 8), [Step::Panic]);
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::Unspecified, 1));
    assert_eq!(answer.request_id, [1]);
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
    let frame = frame(2, 2);
    assert_eq!(
        unleased(bed.ask(frame.clone()).await),
        bed.engines(&frame, 1)
    );
}

/// The `random` picker, which panics at the first pick of a case.
#[derive(Debug)]
struct PanicsOnce(Arc<AtomicBool>);

impl Lifecycle for PanicsOnce {}

impl Picker for PanicsOnce {
    fn name(&self) -> &'static str {
        "panics-once"
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
        let picked_before = self.0.swap(true, Ordering::Relaxed);
        assert!(picked_before, "the case's own panic, inside a stage");
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// A stage that panics inside a decision is held by the engine's own
/// catch, which refuses the request. Its decider answers it `Busy`
/// with no reason and counts it as a fault, as one its own catch held,
/// files no lease, and decides the next request.
#[tokio::test]
async fn a_stage_that_panics_in_a_decision_is_answered_busy_and_counted_as_a_fault() {
    let fired = Arc::new(AtomicBool::new(false));
    let spec = picking("panics-once", move || -> Box<dyn Picker> {
        Box::new(PanicsOnce(Arc::clone(&fired)))
    });
    let mut bed = bed(Some(spec), true, (1, 8), []);
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::Unspecified, 1));
    assert_eq!(answer.request_id, [1]);
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.live(GATEWAY), 0, "no lease was filed");
    let answer = bed.ask(frame(2, 2)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
}

/// A panic's payload can panic at its own drop. The decider frees it
/// under a catch of its own: the request is answered, its thread stands
/// and the deciders tell nothing.
#[tokio::test]
async fn a_panic_whose_payload_panics_at_its_drop_costs_its_own_request_alone() {
    let spec = Some(preset("random", 1));
    let mut bed = bed(spec, true, (1, 8), [Step::Bomb]);
    let ended = bed.deciders.ended();
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::Unspecified, 1));
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
    let frame = frame(2, 2);
    assert_eq!(
        unleased(bed.ask(frame.clone()).await),
        bed.engines(&frame, 1)
    );
    assert!(!*ended.borrow(), "the thread stands");
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

/// Every worker a pick names, its alternates with it.
fn targets(answer: &proto::DecideAnswer) -> Vec<WorkerId> {
    match outcome(answer) {
        Outcome::Pick(pick) => pick.targets.iter().map(|t| WorkerId(t.id)).collect(),
        other => panic!("a pick: {other:?}"),
    }
}

/// How many times each worker of `picked` stands in it, in the
/// workers' order.
fn turns(mut picked: Vec<WorkerId>) -> Vec<usize> {
    picked.sort_unstable();
    picked
        .chunk_by(|one, next| one == next)
        .map(<[WorkerId]>::len)
        .collect()
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

/// One ready member that speaks HTTP alone, asked for under each thing
/// a gateway states of itself. A gateway that states no wire, or one
/// the member speaks, is answered the member. One whose wires the
/// member shares none of is shed under the wires' stage, and so is one
/// that states only a wire this build does not know. A hint of 0 is a
/// decision over no candidates, shed under the hint's stage whatever
/// the wires, and a positive hint cuts nothing and names no stage: the
/// wires that share none are the stage under one. Then a member the
/// gateway cannot dial and a second one it can, beside the first, which
/// it just failed to reach: the second is its whole answer.
#[tokio::test]
async fn a_gateway_is_answered_no_member_it_cannot_dial_and_none_under_a_hint_of_0() {
    const SPOKEN: &str = "model-http";
    let mut bed = leading([]);
    let store = Arc::clone(&bed.store);
    let first = ready(
        &store,
        &bed.reads.objects,
        "h0",
        SPOKEN,
        Role::Single,
        WireModes::HTTP,
    );
    let answer = bed.ask(frame_for(SPOKEN, 100, 100)).await;
    assert_eq!(targets(&answer), [first], "no constraints");
    // The wires a gateway states, its hint, and the stage its decision
    // is shed under where it is shed.
    let cells = [
        (0b001, None, None),
        (0b011, None, None),
        (0b010, None, Some("wires")),
        (0b110, None, Some("wires")),
        (0b1000, None, Some("wires")),
        (0b010, Some(1), Some("wires")),
        (0b001, Some(0), Some("subset-hint")),
        (0b010, Some(0), Some("subset-hint")),
        (0, Some(0), Some("subset-hint")),
        (0b001, Some(1), None),
        (0, Some(7), None),
    ];
    for (id, (wires, hint, stage)) in (1..).zip(cells) {
        let frame = constrained(frame_for(SPOKEN, id, u64::from(id)), wires, hint);
        let answer = bed.ask(frame).await;
        let cell = format!("wires {wires:#b}, a hint of {hint:?}");
        match stage {
            Some(stage) => assert_eq!(
                outcome(&answer),
                &declined(proto::ShedReason::NoEndpoints, stage),
                "{cell}"
            ),
            None => assert_eq!(targets(&answer), [first], "{cell}"),
        }
    }
    // The wires took the member in four decisions: under a hint of 0
    // they had none to take.
    assert_eq!(bed.counts.undialed(), 4);
    ready(
        &store,
        &bed.reads.objects,
        "g0",
        SPOKEN,
        Role::Single,
        WireModes::GRPC,
    );
    let second = ready(
        &store,
        &bed.reads.objects,
        "h1",
        SPOKEN,
        Role::Single,
        WireModes::HTTP,
    );
    for seed in 0..32_u8 {
        let frame = frame_for(SPOKEN, 40 + seed, u64::from(seed));
        let mut frame = constrained(frame, 0b001, None);
        frame.constraints.get_or_insert_default().excluded = vec![first.0];
        let answer = bed.ask(frame).await;
        assert_eq!(targets(&answer), [second], "seed {seed}");
    }
    assert_eq!(bed.counts.undialed(), 4 + 32);
}

/// A gateway that states its wires is answered members that share one
/// with it, its alternates too, and each one its wires took is counted.
/// A member that states no wire is no candidate of a gateway that
/// states some: silence is no waiver. A gateway that states none is not
/// narrowed, and a positive hint, which no stage reads yet, cuts
/// nothing.
#[tokio::test]
async fn a_gateways_wires_narrow_its_decision_to_the_members_that_share_one() {
    let mut bed = leading([]);
    // The fleet's three members state no wire.
    let answer = bed.ask(constrained(frame(1, 1), 0b111, None)).await;
    let shed = declined(proto::ShedReason::NoEndpoints, "wires");
    assert_eq!(outcome(&answer), &shed);
    assert_eq!(bed.counts.undialed(), 3);
    let store = Arc::clone(&bed.store);
    let both = WireModes::HTTP.union(WireModes::GRPC);
    let [_, grpc, either] = [
        ("w0", WireModes::HTTP),
        ("w1", WireModes::GRPC),
        ("w2", both),
    ]
    .map(|(name, wires)| ready(&store, &bed.reads.objects, name, MODEL, Role::Single, wires));
    let mut dialed = [grpc, either];
    dialed.sort_unstable();
    let mut named = BTreeSet::new();
    for (id, seed) in (2..=33).zip(100..) {
        let answer = bed.ask(constrained(frame(id, seed), 0b010, None)).await;
        named.extend(targets(&answer));
    }
    assert_eq!(named.into_iter().collect::<Vec<_>>(), dialed);
    // The member that speaks HTTP alone, in each of the 32 decisions.
    assert_eq!(bed.counts.undialed(), 3 + 32);
    for (id, hint) in (40..).zip([None, Some(1), Some(2), Some(64)]) {
        let answer = bed.ask(constrained(frame(id, 40), 0, hint)).await;
        assert_eq!(targets(&answer).len(), 3, "a hint of {hint:?}");
    }
    assert_eq!(bed.counts.undialed(), 3 + 32);
}

/// The rows a gateway's wires leave are a list of their own and turn a
/// rotation of their own. Sixteen members speak HTTP alone and sixteen
/// gRPC alone under `round_robin`, and the asks alternate between the
/// two wires: each list is sixteen rows cut out of one build, and every
/// member is the pick once in its list's sixteen asks. Under one
/// rotation between the two each list was drawn at every other row.
/// Wires that take no row leave the build's own list and its rotation:
/// asks that state the one wire a pool speaks, in turn with asks that
/// state none, walk its sixteen rows once.
#[tokio::test]
async fn the_rows_a_gateways_wires_leave_turn_a_rotation_of_their_own() {
    const TWO: &str = "model-two-wires";
    const ONE: &str = "model-one-wire";
    let mut bed = bed(Some(preset("round_robin", 1)), true, (2, 8), []);
    let store = Arc::clone(&bed.store);
    for row in 0..16 {
        ready(
            &store,
            &bed.reads.objects,
            &format!("h{row}"),
            TWO,
            Role::Single,
            WireModes::HTTP,
        );
        ready(
            &store,
            &bed.reads.objects,
            &format!("g{row}"),
            TWO,
            Role::Single,
            WireModes::GRPC,
        );
        ready(
            &store,
            &bed.reads.objects,
            &format!("o{row}"),
            ONE,
            Role::Single,
            WireModes::HTTP,
        );
    }
    let mut picked = Vec::new();
    for ask in 0..64_u8 {
        let wires = if ask.is_multiple_of(2) { 0b001 } else { 0b010 };
        let frame = constrained(frame_for(TWO, ask, u64::from(ask)), wires, None);
        picked.push(targets(&bed.ask(frame).await)[0]);
    }
    assert_eq!(turns(picked), [2; 32], "two wires over one pool");
    let mut picked = Vec::new();
    for ask in 64..80_u8 {
        let frame = frame_for(ONE, ask, u64::from(ask));
        let frame = if ask.is_multiple_of(2) {
            constrained(frame, 0b001, None)
        } else {
            frame
        };
        picked.push(targets(&bed.ask(frame).await)[0]);
    }
    assert_eq!(turns(picked), [1; 16], "wires that take no row");
}

/// The budget is judged again where the answer is handed over, at the
/// read the fence takes there and with no clock read more. A decision
/// with a microsecond left leaves. One whose budget ended while it was
/// made leaves as `Busy` under the reason a late arrival is told, and
/// is counted apart from one. The case's budget is half the dispatch
/// span, so no lease's span ends where its decision's budget does.
#[tokio::test]
async fn a_decision_whose_budget_ended_while_it_was_made_leaves_as_busy_and_is_counted_apart() {
    let budget = DISPATCH_TTL / 2;
    let halved = |id, seed| budgeted(frame(id, seed), budget);
    let short = budget - Duration::from_micros(1);
    let under = budget - Duration::from_nanos(999);
    let steps = [
        Duration::ZERO,
        short,
        Duration::ZERO,
        under,
        Duration::ZERO,
        budget,
        budget,
    ]
    .map(Step::At);
    let mut bed = leading(steps);
    let answer = bed.ask(halved(1, 1)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 0);
    // Under one microsecond left is gone, as where a request is taken.
    for (id, count) in [(2, 1), (3, 2)] {
        let answer = bed.ask(halved(id, 2)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!((answer.request_id.as_slice(), answer.term), (&[id][..], 1));
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), count);
    }
    // A request that arrived late is told the same reason at its
    // decider's first read, and is none of the count.
    let answer = bed.ask(halved(4, 4)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 2);
    assert_eq!(bed.script.readers.lock().len(), 7);
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

/// The bed's frame under a session key in the clear.
fn keyed_in_the_clear(key: &str, seed: u64) -> proto::DecideRequest {
    let mut frame = frame(u8::try_from(seed % 251).expect("under 251"), seed);
    frame
        .facts
        .as_mut()
        .expect("the case's frame has facts")
        .session_key = Some(proto::Identity {
        form: Some(proto::identity::Form::Plain(key.to_owned())),
    });
    frame
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

/// A keyed session keeps its worker across an apply of a new
/// generation. The store's engine keeps one session table for its
/// life, and each generation's stages read it
/// (`AppliedSpec::share_sessions`): a decision of a dispatched key is
/// made on its pin under either generation.
#[tokio::test]
async fn a_keyed_session_keeps_its_worker_across_an_apply() {
    use crate::{
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::{self, tests::a_dispatched},
    };
    const KEYS: u64 = 24;
    let mut bed = bed(Some(preset("manual", 1)), true, (1, 8), []);
    let at = bed.script.taken;
    let verdict = |bed: &Bed, answer: &proto::DecideAnswer| {
        LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
            .expect("the lease is on file")
            .lease
            .session_verdict
    };
    // The key's value as its decision made it, read off the lease.
    let session = |bed: &Bed, answer: &proto::DecideAnswer| {
        LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
            .expect("the lease is on file")
            .lease
            .session
    };
    let pins = |bed: &Bed, session: view::Session| {
        bed.store
            .deciding(|world| world.spec().engine().sessions().pins(session))
    };
    // Each key's first decision claims a worker, and its gateway's
    // Dispatched frame commits it: the engine's pin.
    let mut first = Vec::new();
    for key in 1..=KEYS {
        let answer = bed.ask(keyed(key, 1_000 + key)).await;
        let worker = picked(&answer);
        assert_eq!(verdict(&bed, &answer), SessionVerdict::Claimed(worker));
        let hand = Hand::new(GATEWAY);
        lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&answer).id), at)
            .expect("the frame reads");
        first.push((key, worker, session(&bed, &answer)));
    }
    // The keys stand on more than one worker: a bed that kept every
    // key on one would hold nothing.
    let stood = bed
        .workers
        .iter()
        .filter(|member| first.iter().any(|&(_, worker, _)| worker == **member))
        .count();
    assert!(stood > 1, "the keys stand on {stood} worker");
    for &(key, worker, _) in &first {
        let answer = bed.ask(keyed(key, 2_000 + key)).await;
        assert_eq!(picked(&answer), worker);
        assert_eq!(verdict(&bed, &answer), SessionVerdict::Pinned);
    }
    // A new generation: the pins of the one before stand.
    let held = pins(&bed, first[0].2);
    assert!(held.is_some());
    bed.store
        .apply(preset("manual", 2))
        .expect("a later generation");
    assert_eq!(pins(&bed, first[0].2), held);
    for &(key, worker, session) in &first {
        let answer = bed.ask(keyed(key, 3_000 + key)).await;
        assert_eq!(
            picked(&answer),
            worker,
            "key {key} left its worker at the apply"
        );
        assert_eq!(verdict(&bed, &answer), SessionVerdict::Pinned);
        assert_eq!(pins(&bed, session), Some([worker, WorkerId(0)]));
    }
}

/// The controller's own service counts a key's requests. Under a rule
/// with an in-flight cap the request that finds the cap met on its
/// key's worker is placed again and lands on a worker with fewer
/// requests. The pin moves there where that request is decided, ahead
/// of its gateway's Dispatched frame, so the key's next request follows
/// it, and the one after stays once one of the two ended. The frames
/// carry the key as its hash.
#[tokio::test]
async fn the_service_hands_a_key_on_at_its_rules_cap() {
    handed_on_at_the_cap(|seed| keyed(1, seed)).await;
}

/// The same of a key in the clear, the form a gateway sends a key it
/// holds as text in. The lease carries the session's number alone, so
/// the count stands under that number whichever form the frame took.
#[tokio::test]
async fn the_service_hands_a_key_in_the_clear_on_at_its_rules_cap() {
    handed_on_at_the_cap(|seed| keyed_in_the_clear("talk-1", seed)).await;
}

/// One key's five requests under a rule whose in-flight cap is two,
/// each made by `frame` from its seed: the first claims a worker, the
/// second stays, the third is placed again and moves the pin, the
/// fourth follows the move, and the fifth stays behind an end.
async fn handed_on_at_the_cap(frame: impl Fn(u64) -> proto::DecideRequest) {
    use crate::{
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::{self, tests::a_dispatched},
    };
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    rule.inflight_cap = 2;
    // The rule places a key on the worker with the fewest requests: a
    // request placed again at the cap leaves the key's worker.
    rule.assignment = crate::document::Assignment::MinLoad;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let at = bed.script.taken;
    let verdict = |bed: &Bed, answer: &proto::DecideAnswer| {
        LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
            .expect("the lease is on file")
            .lease
            .session_verdict
    };
    // The key's first request claims a worker, and its gateway's
    // Dispatched frame pins the key there.
    let first = bed.ask(frame(1_001)).await;
    let owner = picked(&first);
    assert_eq!(verdict(&bed, &first), SessionVerdict::Claimed(owner));
    let hand = Hand::new(GATEWAY);
    lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&first).id), at)
        .expect("the frame reads");
    // One request of the key in flight: the second stays.
    let second = bed.ask(frame(1_002)).await;
    assert_eq!(picked(&second), owner);
    assert_eq!(verdict(&bed, &second), SessionVerdict::Pinned);
    // Two in flight on the key's worker: the third is placed again,
    // on a worker with fewer requests. Its decision moves the pin
    // there, and no Dispatched frame has come for it.
    let third = bed.ask(frame(1_003)).await;
    let moved = picked(&third);
    assert_ne!(moved, owner);
    assert_eq!(verdict(&bed, &third), SessionVerdict::Respill);
    // The fourth follows the move, as a gateway's sticky map sends it:
    // that worker carries two of the key's requests now.
    let fourth = bed.ask(frame(1_004)).await;
    assert_eq!(picked(&fourth), moved);
    assert_eq!(verdict(&bed, &fourth), SessionVerdict::Pinned);
    // One of the two ended: the key's next request is under the cap.
    let ended = LeaseId(lease_of(&third).id);
    let gone = ReleaseReason::ClientGone;
    assert!(LeasesStore::released(
        &*bed.store,
        GATEWAY,
        ended,
        None,
        gone
    ));
    let fifth = bed.ask(frame(1_005)).await;
    assert_eq!(picked(&fifth), moved);
    assert_eq!(verdict(&bed, &fifth), SessionVerdict::Pinned);
}

/// The keys the bed's worker objects count, over all of them: a key
/// counts once on each worker that carries a request of it.
fn keys_counted(bed: &Bed) -> usize {
    let objects = bed.reads.objects.registry().get_all();
    let keys = objects.iter().map(|object| object.routing_key_load());
    keys.sum()
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

/// The service counts a keyed single request for its key under any
/// sessions rule and under none: its lease counts on the worker object
/// it books, as a gateway's load guard counts every request that
/// carries a key (`crates/worker/src/load/inflight.rs:196-199` at main
/// `5e609dc2d`). The stock rule's cap is disabled, and the `random`
/// preset has no sessions rule. A request with no key counts for none.
#[tokio::test]
async fn the_service_counts_a_keyed_request_under_any_rule_and_under_none() {
    let capped = {
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").inflight_cap = 2;
        applied(&document)
    };
    let ruleless = PolicySpec::preset("random").expect("a preset of this build");
    assert!(ruleless
        .bindings
        .iter()
        .all(|binding| binding.sessions.is_none()));
    let documents = [
        ("a rule with no cap", preset("manual", 1)),
        ("a rule with a cap", capped),
        ("no rule", applied(&ruleless)),
    ];
    for (under, document) in documents {
        let mut bed = bed(Some(document), true, (1, 8), []);
        bed.ask(frame(1, 1_000)).await;
        assert_eq!(keys_counted(&bed), 0, "{under}: a request with no key");
        bed.ask(keyed(1, 1_001)).await;
        assert_eq!(keys_counted(&bed), 1, "{under}: a keyed request");
    }
}

/// A rule that places a new key on the worker with the fewest keys
/// reads the keys of the leases the service holds, under no cap. One
/// worker carries three keys' requests and the others none: no new key
/// takes it. Each new key's count leaves at its lease's release, so
/// the others stay at none. The gateway's classic sticky policy
/// answers the same: a new key takes a worker with the fewest keys in
/// flight (`model_gateway/src/policies/manual.rs:479-480` at main
/// `5e609dc2d`), which its load guard counts for every request that
/// carries a key (`crates/worker/src/load/inflight.rs:196-199`).
#[tokio::test]
async fn the_service_places_a_new_key_by_the_keys_its_leases_count_under_no_cap() {
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    let rule = document.bindings[0].sessions.as_mut();
    rule.expect("the preset's rule").assignment = crate::document::Assignment::MinGroup;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let (busy, others) = (bed.workers[0], [bed.workers[1], bed.workers[2]]);
    // Three keys' requests on one worker, the only one ready.
    for other in others {
        set_status(&bed, other, WorkerStatus::NotReady);
    }
    for key in 1..=3 {
        let answer = bed.ask(keyed(key, 1_000 + key)).await;
        assert_eq!(picked(&answer), busy);
    }
    for other in others {
        set_status(&bed, other, WorkerStatus::Ready);
    }
    // Each new key takes a worker that carries no key, and its request
    // ends before the next key's.
    for key in 4..=27 {
        let answer = bed.ask(keyed(key, 1_000 + key)).await;
        assert_ne!(picked(&answer), busy, "key {key}");
        let ended = LeaseId(lease_of(&answer).id);
        let gone = ReleaseReason::ClientGone;
        let store: &dyn LeasesStore = bed.store.as_ref();
        assert!(store.released(GATEWAY, ended, None, gone));
    }
    // The three keys of the first worker alone: each new key's count
    // left with its lease.
    assert_eq!(keys_counted(&bed), 3);
}

/// Under a rule with no cap a key's count leaves its worker's object
/// with the key's lease, at a release.
#[tokio::test]
async fn a_key_under_a_rule_with_no_cap_is_counted_until_its_lease_is_released() {
    let mut bed = bed(Some(preset("manual", 1)), true, (1, 8), []);
    let answer = bed.ask(keyed(1, 1_001)).await;
    assert_eq!(keys_counted(&bed), 1);
    let ended = LeaseId(lease_of(&answer).id);
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert!(store.released(GATEWAY, ended, None, ReleaseReason::ClientGone));
    assert_eq!(keys_counted(&bed), 0);
}

/// Under a rule with no cap a key's count leaves its worker's object
/// where no dispatch names its lease: a decision past the lease's
/// dispatch deadline ends it as abandoned.
#[tokio::test]
async fn a_key_under_a_rule_with_no_cap_is_counted_until_its_dispatch_deadline() {
    let mut bed = bed(Some(preset("manual", 1)), true, (1, 8), []);
    bed.ask(keyed(1, 1_001)).await;
    assert_eq!(keys_counted(&bed), 1);
    // A request with no key, decided past that deadline.
    let late = DISPATCH_TTL + Duration::from_secs(1);
    bed.script.steps.lock().push_back(Step::At(late));
    assert_eq!(bed.submit_at(frame(2, 1_002), GATEWAY, late), None);
    bed.answer().await;
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!(store.live(GATEWAY), 1, "the late request's lease alone");
    assert_eq!(keys_counted(&bed), 0);
}

/// A cap that a new document sets counts the key's requests that stood
/// before it. Two requests of a key stand on its worker, decided under
/// a rule with no cap, and a third is decided and dispatched there
/// under the document that sets a cap of two. The key's next request
/// finds three on its pin and is placed again, on another worker.
#[tokio::test]
async fn a_cap_a_new_document_sets_counts_the_requests_that_stood_before_it() {
    use crate::{
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::{self, tests::a_dispatched},
    };
    let ruled = |cap: Option<u16>, generation: u64| {
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = generation;
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        // A request placed again at the cap leaves the key's worker.
        rule.assignment = crate::document::Assignment::MinLoad;
        if let Some(cap) = cap {
            rule.inflight_cap = cap;
        }
        applied(&document)
    };
    let mut bed = bed(Some(ruled(None, 1)), true, (1, 8), []);
    let at = bed.script.taken;
    let (owner, others) = (bed.workers[0], [bed.workers[1], bed.workers[2]]);
    let dispatched = |bed: &Bed, answer: &proto::DecideAnswer| {
        let (hand, frame) = (Hand::new(GATEWAY), a_dispatched(lease_of(answer).id));
        lifecycle::dispatched(&bed.store, &hand, &frame, at).expect("the frame reads");
    };
    // The key's worker is the only one ready until its last request.
    for other in others {
        set_status(&bed, other, WorkerStatus::NotReady);
    }
    // Two requests of the key under the rule with no cap.
    for seed in [1_001, 1_002] {
        let answer = bed.ask(keyed(1, seed)).await;
        assert_eq!(picked(&answer), owner);
        dispatched(&bed, &answer);
    }
    // The cap, and the key's third request under it: its dispatch pins
    // the key on its worker under the new document.
    bed.store
        .apply(ruled(Some(2), 2))
        .expect("a later generation");
    let third = bed.ask(keyed(1, 1_003)).await;
    assert_eq!(picked(&third), owner);
    dispatched(&bed, &third);
    for other in others {
        set_status(&bed, other, WorkerStatus::Ready);
    }
    // Three of the key's requests on its pin, two from before the cap.
    let fourth = bed.ask(keyed(1, 1_004)).await;
    assert_ne!(picked(&fourth), owner);
    let filed = LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(&fourth).id));
    let verdict = filed.expect("the lease is on file").lease.session_verdict;
    assert_eq!(verdict, SessionVerdict::Respill);
}
