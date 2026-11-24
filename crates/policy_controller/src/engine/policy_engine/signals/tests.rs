//! The cases of a document's named signals over worker lists: each decides
//! through the engine's own entry, over worker objects and a load snapshot
//! as a load monitor publishes one, and holds what a stage of the document
//! read to what the snapshot's reports state. No case hands a cell: where a
//! request carries cells, they are a caller's that the engine must not read.

use std::{
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::{
    common_proto,
    identity::{
        basic::{ConnectionMode, Worker},
        builder::BasicWorkerBuilder,
    },
    load::{
        monitor::{poll_group_once_for_test, LoadConsumer, WorkerMonitor},
        state::{LoadReceiver, LoadSnapshot},
    },
    model_card::ModelCard,
    test_support::{no_health_check, ready_worker, Members, WorkerRegistry},
    worker::{
        NamedValue, SchedulerLoadSnapshot, WorkerGroupKey, WorkerLoadResponse, WorkerSpec,
        WorkerStatus, WorkerType,
    },
};
use tokio::{net::TcpListener, task::JoinSet};

use crate::{
    document::spec::{AcrossRanks, AppliedSpec, Needs, PolicySpec, ShadowSpec, SignalSpec},
    engine::{
        decision::{Decision, ShadowKind},
        decision_engine::{DecisionEngine, Request, RequestMaker, WorkerLists},
        framework::{Scorer, TieBand},
        list_cell::tests::PublishedList,
        policy_engine::PolicyEngine,
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, Identity, RequestFacts, WorkerId},
    },
    load::{SignalCell, SignalCells, SignalColumn},
    stages::{registry::StageRegistry, Lifecycle},
    store::GatewayId,
    tracking::{
        lease::{EventKind, EventSet, Lease, Outcome, Side, TransportClass},
        DecisionGuard,
    },
};

mod pair;

const MODEL: &str = "model-a";
/// A model the two-profile document binds to the profile that reads no
/// signal.
const PLAIN_MODEL: &str = "model-b";

/// What a stage read of the signal `held`: each row it was asked about, by
/// the row's place in the list, with the value read there.
type Read = Arc<Mutex<Vec<(u32, Option<f64>)>>>;

/// A scorer of a host's own build that reads the signal `held` of every row
/// it is asked about, and keeps what it read. It answers row by row, so a
/// picker that draws first hands it the rows drawn and no other.
#[derive(Debug)]
struct ReadsHeld {
    held: SignalColumn,
    read: Read,
}

impl Lifecycle for ReadsHeld {}

impl Scorer for ReadsHeld {
    fn name(&self) -> &'static str {
        "reads-held"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let rows = kept.iter().map(|&row| &candidates[row as usize]);
        let read = rows.map(|row| (row.position, self.held.read(facts, row)));
        self.read.lock().extend(read);
        out.fill(0.0);
        TieBand::NORMALIZED
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn needs(&self) -> Needs {
        Needs {
            signal_cells: true,
            ..Needs::default()
        }
    }
}

/// A scorer of a host's own build that reads no signal, and keeps whether
/// the facts of each decision it scored carried signal cells.
#[derive(Debug)]
struct SeesCells(Arc<Mutex<Vec<bool>>>);

impl Lifecycle for SeesCells {}

impl Scorer for SeesCells {
    fn name(&self) -> &'static str {
        "sees-cells"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.0.lock().push(facts.signal_cells.is_some());
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// A scorer of a host's own build that reads no signal and hears the issue
/// of each lease: it keeps the instant each lease was issued at.
#[derive(Debug)]
struct HearsIssue(Arc<Mutex<Vec<u64>>>);

impl Lifecycle for HearsIssue {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push(lease.issued_micros);
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Issued])
    }
}

impl Scorer for HearsIssue {
    fn name(&self) -> &'static str {
        "hears-issue"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// An engine with the three scorers above among its stages, and what each
/// kept.
struct Built {
    engine: PolicyEngine,
    read: Read,
    seen: Arc<Mutex<Vec<bool>>>,
    issued: Arc<Mutex<Vec<u64>>>,
}

/// The engine of `document`, a policy document as JSON.
fn built(document: &str) -> Built {
    let document: PolicySpec = serde_json::from_str(document).expect("a well-formed document");
    built_of(&document)
}

fn built_of(document: &PolicySpec) -> Built {
    let (read, seen) = (Read::default(), Arc::new(Mutex::new(Vec::new())));
    let issued = Arc::new(Mutex::new(Vec::new()));
    let (kept, saw, heard) = (Arc::clone(&read), Arc::clone(&seen), Arc::clone(&issued));
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer_with("reads-held", move |build| {
        Ok(Box::new(ReadsHeld {
            held: build.signal_column("held")?,
            read: Arc::clone(&kept),
        }))
    });
    added.expect("a free name");
    let added = stages.scorer("sees-cells", move || Box::new(SeesCells(Arc::clone(&saw))));
    added.expect("a free name");
    let added = stages.scorer("hears-issue", move || {
        Box::new(HearsIssue(Arc::clone(&heard)))
    });
    added.expect("a free name");
    let spec = AppliedSpec::validate_in(document, &stages, &[]);
    Built {
        engine: PolicyEngine::new(spec.expect("the stages hold the three scorers")),
        read,
        seen,
        issued,
    }
}

/// A document of one signal, `held`, in tokens, read by its one profile's
/// scorer. `signal` states the signal beyond its unit, and `pick` names the
/// profile's picker.
fn reads(signal: &str, pick: &str) -> String {
    format!(
        r#"{{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{{"held":{{"unit":"tokens"{signal}}}}},
            "profiles":{{"p":{{"score":["reads-held"],"pick":"{pick}"}}}}}}"#
    )
}

/// The rule of the two token counts a report offers by name: a worker has
/// room where what it uses, with the request's own tokens, is within its
/// capacity.
const ROOM: &str = r#"{"has-room":{
    "used":{"signal.kv_tokens_used":1,"request.prompt_tokens":1},
    "capacity":"signal.kv_tokens_capacity"}}"#;

fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&url(at), MODEL))
        .collect()
}

/// One value a report states by name, with no age of its own.
fn stated(name: &str, unit: &str, value: f64) -> NamedValue {
    NamedValue {
        name: name.into(),
        unit: unit.into(),
        value,
        age_ms: None,
    }
}

/// The signal `held` stated in tokens.
fn held(value: f64) -> Vec<NamedValue> {
    vec![stated("held", "tokens", value)]
}

/// A report of as many ranks as `ranks` holds lists, each rank stating its
/// list by name.
fn ranks(ranks: Vec<Vec<NamedValue>>) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: ranks.len() as i32,
        loads: (0..)
            .zip(ranks)
            .map(|(dp_rank, values)| SchedulerLoadSnapshot {
                dp_rank,
                values,
                ..SchedulerLoadSnapshot::default()
            })
            .collect(),
        ..WorkerLoadResponse::default()
    }
}

/// A report of one rank that states `values` by name.
fn report(values: Vec<NamedValue>) -> WorkerLoadResponse {
    ranks(vec![values])
}

/// A report of one rank that uses `used` of `capacity` tokens: the two
/// counts a report offers by name, with no list of named values.
fn holds(used: i32, capacity: i32) -> WorkerLoadResponse {
    let mut report = report(vec![]);
    for rank in &mut report.loads {
        rank.num_used_tokens = used;
        rank.max_total_num_tokens = capacity;
    }
    report
}

/// A load snapshot as a load monitor publishes one: each report of
/// `reports` under the address of the worker at its place in the list.
fn snapshot(reports: impl IntoIterator<Item = (usize, WorkerLoadResponse)>) -> Arc<LoadSnapshot> {
    let reports = reports.into_iter().map(|(at, report)| (url(at), report));
    LoadSnapshot::from_loads_for_test(reports.collect())
}

/// A snapshot of `workers` workers, each stating `held` at `base` more than
/// its place in the list.
fn stepped(workers: u32, base: f64) -> Arc<LoadSnapshot> {
    snapshot((0..workers).map(|at| (at as usize, report(held(base + f64::from(at))))))
}

/// What a stage reads of `held` under [`stepped`], over four workers.
fn stepped_values(base: f64) -> [(u32, Option<f64>); 4] {
    [0, 1, 2, 3].map(|at| (at, Some(base + f64::from(at))))
}

/// A snapshot of `workers` workers in which the worker at `fits` alone
/// has room for a request.
fn room_at(workers: usize, fits: usize) -> Arc<LoadSnapshot> {
    snapshot((0..workers).map(|at| (at, holds(if at == fits { 100 } else { 2_000 }, 1_000))))
}

/// The worker object of one rank of a data-parallel worker of two ranks: a
/// row of its own in a list, under the worker's address and its rank.
fn rank_row(rank: usize) -> Arc<dyn Worker> {
    let worker = <BasicWorkerBuilder>::new("http://ranked:8000")
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .model(ModelCard::new(MODEL))
        .dp_config(rank, 2)
        .health_config(no_health_check())
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// A ready worker object at `url` that stands for `ranks` ranks: its
/// registration states that width and no rank of its own.
fn wide_at(url: &str, ranks: usize) -> Arc<dyn Worker> {
    let mut spec = WorkerSpec::new(url);
    spec.dp_size = Some(ranks);
    let worker = <BasicWorkerBuilder>::from_spec(spec)
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .model(ModelCard::new(MODEL))
        .health_config(no_health_check())
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// [`wide_at`] the address of the list's place `at`.
fn wide(at: usize, ranks: usize) -> Arc<dyn Worker> {
    wide_at(&url(at), ranks)
}

/// A polled engine at an address of its own, served for as long as the
/// set that comes with the address is held: its load route answers
/// `answer`, and it serves no other route.
async fn polled_engine(answer: &'static str) -> (String, JoinSet<std::io::Result<()>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
    let url = format!("http://{}", listener.local_addr().expect("its address"));
    let route = axum::routing::get(move || async move { answer });
    let routes = axum::Router::new().route("/v1/loads", route);
    let mut served = JoinSet::new();
    served.spawn(async move { axum::serve(listener, routes).await });
    (url, served)
}

/// The readers of a case's load monitor: the case's decisions, which read
/// the snapshot it publishes and book nothing through it.
struct Readers;

impl LoadConsumer for Readers {
    fn needs_loads(&self) -> bool {
        true
    }

    fn reconcile_in_flight(&self, _: &dyn Worker) {}
}

/// The worker crate's own load monitor over `workers`, each registered
/// with it, and the channel it publishes its snapshots on. The cases that
/// push records hand them to this monitor, so their reports are the ones
/// the worker crate makes of pushed records, and no report built here.
fn monitor_over(workers: &[Arc<dyn Worker>]) -> (Arc<WorkerMonitor<Members>>, LoadReceiver) {
    let registry = Arc::new(WorkerRegistry::new());
    for worker in workers {
        let registered = registry.register(Arc::clone(worker));
        registered.expect("an address of its own");
    }
    let monitor = Arc::new(WorkerMonitor::new(
        registry,
        Arc::new(Readers),
        1,
        false,
        false,
    ));
    let published = monitor.subscribe();
    (monitor, published)
}

/// A record a worker pushes for one of its ranks: it states `held` in
/// tokens, with no age of its own.
fn pushes(value: f64) -> common_proto::EngineLoad {
    let held = common_proto::NamedValue {
        name: "held".into(),
        unit: "tokens".into(),
        value,
        age_ms: None,
    };
    common_proto::EngineLoad {
        values: vec![held],
        ..Default::default()
    }
}

/// The snapshot `monitor` publishes of the records pushed so far.
fn published(monitor: &Arc<WorkerMonitor<Members>>, channel: &LoadReceiver) -> Arc<LoadSnapshot> {
    monitor.flush_pushed_for_test();
    Arc::clone(&channel.borrow())
}

/// A request stated ahead over one list, with the snapshot its host gives
/// it, where it gives one.
struct Asks<'a> {
    request: Request<'a>,
    members: &'a PublishedList,
    loads: Option<Arc<LoadSnapshot>>,
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

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads.as_ref()
    }
}

/// A request of `seed` over `members` that gives `loads`.
fn asks<'a>(seed: u64, members: &'a PublishedList, loads: Option<&Arc<LoadSnapshot>>) -> Asks<'a> {
    Asks {
        request: Request {
            id: b"request",
            facts: RequestFacts {
                model: MODEL,
                seed,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        },
        members,
        loads: loads.cloned(),
    }
}

/// The whole decision of `asks`' request at `now`.
fn decided_at(engine: &PolicyEngine, asks: &Asks<'_>, now: Instant) -> Decision {
    let decided = engine.decide_under(GatewayId::LOCAL, asks, now);
    let whole = decided.expect("an answer").whole();
    whole.expect("a whole decision").0.decision
}

/// The place in its list of the worker a decision names first.
fn first_of(decision: &Decision) -> u32 {
    let pick = decision.pick().expect("a pick");
    pick.targets.first().expect("a target").position
}

/// What the stage read of `held` at the one decision of `asks`' request at
/// `now`, in the list's order.
fn read_at(built: &Built, asks: &Asks<'_>, now: Instant) -> Vec<(u32, Option<f64>)> {
    built.read.lock().clear();
    let _ = decided_at(&built.engine, asks, now);
    let mut read = mem::take(&mut *built.read.lock());
    read.sort_by_key(|&(place, _)| place);
    read
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

/// Each worker's value is the one its own report states, read at the
/// worker's place in the list. A worker whose report leaves the name out
/// reads no value, and so does a worker the snapshot holds no report of.
/// The host hands the snapshot and no cell.
#[test]
fn each_workers_stated_value_is_read_at_its_place_in_the_list() {
    let built = built(&reads("", "best"));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let both = vec![
        stated("other", "tokens", 1.0),
        stated("held", "tokens", 20.0),
    ];
    let loads = snapshot([
        (0, report(held(10.0))),
        (1, report(both)),
        (2, report(vec![stated("other", "tokens", 30.0)])),
    ]);
    let read = read_at(&built, &asks(7, &members, Some(&loads)), Instant::now());
    assert_eq!(
        read,
        [(0, Some(10.0)), (1, Some(20.0)), (2, None), (3, None)]
    );
}

/// Cells a caller put on its request are not read: they are by the
/// caller's own positions, and the engine reads the snapshot its decision
/// holds. Here the caller's cells hold a value at every place. The stage
/// reads the one value the snapshot states, and under no snapshot it reads
/// no value at all.
#[test]
fn cells_a_caller_put_on_the_request_are_not_read() {
    let built = built(&reads("", "best"));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let callers = SignalCell {
        value: 7.0,
        sampled_at_micros: 0,
    };
    let cells = SignalCells::new(vec![callers; 4], 1);
    let loads = snapshot([(2, report(held(10.0)))]);
    let mut hosted = asks(7, &members, Some(&loads));
    hosted.request.facts.signal_cells = Some(&cells);
    let read = read_at(&built, &hosted, Instant::now());
    assert_eq!(read, [(0, None), (1, None), (2, Some(10.0)), (3, None)]);
    hosted.loads = None;
    let read = read_at(&built, &hosted, Instant::now());
    assert_eq!(read, [(0, None), (1, None), (2, None), (3, None)]);
}

/// A value is read where a decision may read it: stated under the unit the
/// document declares, byte for byte, finite, and of magnitude at most
/// 1e15. Any other statement reads as no value.
#[test]
fn a_value_under_another_unit_or_out_of_range_is_not_read() {
    let built = built(&reads("", "best"));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let under = |unit: &str, value: f64| report(vec![stated("held", unit, value)]);
    let loads = snapshot([
        (0, under("tokens", 1e15)),
        (1, under("tokens", -1e15)),
        (2, under("tokens", 1e15 + 1.0)),
        (3, under("tokens", f64::INFINITY)),
        (4, under("tokens", f64::NAN)),
        (5, under("Tokens", 5.0)),
        (6, under("bytes", 6.0)),
        (7, under("tokens", 7.0)),
    ]);
    let read = read_at(&built, &asks(7, &members, Some(&loads)), Instant::now());
    let values: Vec<_> = read.iter().map(|&(_, value)| value).collect();
    let none = None;
    assert_eq!(
        values,
        [
            Some(1e15),
            Some(-1e15),
            none,
            none,
            none,
            none,
            none,
            Some(7.0)
        ]
    );
}

/// A value ends by its signal's maximum age, counted from its own sample:
/// the report's sample instant, less the age the report states for the
/// value. At exactly that age it is read, and a millisecond past it, it is
/// not.
#[test]
fn a_value_ends_by_its_signals_maximum_age_from_its_own_sample() {
    let built = built(&reads(r#","max_age_ms":500"#, "best"));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let sampled = Instant::now();
    let at = |values| WorkerLoadResponse {
        sampled_at: Some(sampled),
        ..report(values)
    };
    // The second worker's value was 200 ms old when its report was made.
    let aged = NamedValue {
        age_ms: Some(200),
        ..stated("held", "tokens", 20.0)
    };
    let loads = snapshot([(0, at(held(10.0))), (1, at(vec![aged]))]);
    let asks = asks(7, &members, Some(&loads));
    for (millis, first, second) in [
        (300, Some(10.0), Some(20.0)),
        (301, Some(10.0), None),
        (500, Some(10.0), None),
        (501, None, None),
    ] {
        let read = read_at(&built, &asks, sampled + Duration::from_millis(millis));
        assert_eq!(
            read,
            [(0, first), (1, second)],
            "{millis} ms behind the report's sample"
        );
    }
}

/// A report that lists every rank it counts gives one value, made of its
/// ranks' values as the signal declares across ranks, however the report
/// was made. A report short of ranks gives none. Here the first three
/// workers' objects each stand for two ranks. The first worker's
/// report lists two ranks, as a polled answer does. The second's is the
/// report a poll publishes while both its ranks push: the poll's own
/// version, both ranks' entries. It counts no rank, as an answer that
/// leaves its count out does, and lists two: it is not short. The third's
/// lists one rank and counts two, as a partial answer does: the rank it
/// leaves out states nothing, and no value is made. The fourth's is one
/// rank's whole.
#[test]
fn a_whole_report_of_several_ranks_gives_one_value_and_a_short_one_none() {
    let made = [
        ("sum", 40.0, 12.0),
        ("max", 30.0, 7.0),
        ("min", 10.0, 5.0),
        ("mean", 20.0, 6.0),
    ];
    for (across, listed, published) in made {
        let signal = format!(r#","max_age_ms":500,"across_ranks":"{across}""#);
        let built = built(&reads(&signal, "best"));
        let one = ready_worker(&url(3), MODEL);
        let list: Arc<[Arc<dyn Worker>]> = [wide(0, 2), wide(1, 2), wide(2, 2), one].into();
        let members = PublishedList::new(Arc::clone(&list));
        let now = Instant::now();
        let merged = WorkerLoadResponse {
            version: "1".into(),
            dp_rank_count: 0,
            sampled_at: Some(now),
            ..ranks(vec![held(5.0), held(7.0)])
        };
        let partial = WorkerLoadResponse {
            dp_rank_count: 2,
            ..report(held(10.0))
        };
        let loads = snapshot([
            (0, ranks(vec![held(10.0), held(30.0)])),
            (1, merged),
            (2, partial),
            (3, report(held(10.0))),
        ]);
        let read = read_at(&built, &asks(7, &members, Some(&loads)), now);
        assert_eq!(
            read,
            [
                (0, Some(listed)),
                (1, Some(published)),
                (2, None),
                (3, Some(10.0))
            ],
            "{across}"
        );
    }
}

/// A report of several ranks is read where it lists as many ranks as its
/// worker object stands for: the width the object's registration states,
/// with no rank of its own. A pushed report counts the rows it holds, so
/// its own count does not say that a rank is still to report, and every
/// report here lists each rank it counts. The first worker's object stands
/// for two ranks and its report lists two: one value, their sum. The
/// second's states no width and the third's registers a rank of two: each
/// stands for one rank, and a report of two is not read. The fourth's
/// stands for two and its report lists three. The fifth's stands for three
/// and its report lists two. Neither is read. The sixth's stands for two
/// and its report lists one rank: a report of one row is read as the
/// worker's whole, whatever its object stands for.
#[test]
fn a_report_of_several_ranks_is_read_of_the_object_that_stands_for_as_many() {
    let built = built(&reads(r#","across_ranks":"sum""#, "best"));
    let list: Arc<[Arc<dyn Worker>]> = [
        wide(0, 2),
        ready_worker(&url(1), MODEL),
        rank_row(1),
        wide(3, 2),
        wide(4, 3),
        wide(5, 2),
    ]
    .into();
    let members = PublishedList::new(Arc::clone(&list));
    let two = || ranks(vec![held(10.0), held(30.0)]);
    let three = ranks(vec![held(1.0), held(2.0), held(4.0)]);
    let stated = [two(), two(), two(), three, two(), report(held(9.0))];
    let reports = list.iter().map(|worker| worker.url().to_owned());
    let loads = LoadSnapshot::from_loads_for_test(reports.zip(stated).collect());
    let read = read_at(&built, &asks(7, &members, Some(&loads)), Instant::now());
    let whole = [Some(40.0), None, None, None, None, Some(9.0)];
    assert_eq!(read, (0..).zip(whole).collect::<Vec<_>>());
}

/// What a polled engine of three ranks answers on its load route: each
/// rank states `held` in tokens.
const THREE_RANKS: &str = r#"{"dp_rank_count":3,"loads":[
    {"dp_rank":0,"values":[{"name":"held","unit":"tokens","value":5.0}]},
    {"dp_rank":1,"values":[{"name":"held","unit":"tokens","value":7.0}]},
    {"dp_rank":2,"values":[{"name":"held","unit":"tokens","value":9.0}]}]}"#;

/// A worker object that stands for three ranks, two of which push their
/// load, through the worker crate's own monitor. The monitor's file of
/// pushed records counts the two rows it holds, and no value is made of
/// them: the object stands for three. Then the two ranks go quiet and the
/// monitor polls the engine, whose answer lists all three ranks: the
/// poll's report reads one value, their sum. The poll adds no rank to the
/// file, so the worker's next pushed record publishes two rows under a
/// count of two again, and again no value is read. Read by the report's
/// own count alone, those two rows would stand for the whole worker.
#[tokio::test]
async fn a_pushed_report_short_of_its_objects_ranks_gives_no_value_behind_a_whole_poll() {
    const INTERVAL: Duration = Duration::from_secs(10);
    let built = built(&reads(r#","across_ranks":"sum""#, "best"));
    let (engine, _served) = polled_engine(THREE_RANKS).await;
    let list: Arc<[Arc<dyn Worker>]> = [wide_at(&engine, 3)].into();
    let members = PublishedList::new(Arc::clone(&list));
    let (monitor, channel) = monitor_over(&list);
    let stated = |loads: &LoadSnapshot| {
        let report = loads.get(&engine).expect("the worker's report");
        (report.loads.len(), report.dp_rank_count)
    };
    let read = |loads: &Arc<LoadSnapshot>, now| {
        let read = read_at(&built, &asks(7, &members, Some(loads)), now);
        read.first().expect("the one worker").1
    };
    // The two ranks' records are an interval old at the tick, so the tick
    // polls the worker.
    let tick = Instant::now();
    let pushed_at = tick.checked_sub(INTERVAL).expect("an older clock");
    monitor.apply_pushed_load(&list[0], 0, &pushes(50.0), pushed_at);
    monitor.apply_pushed_load(&list[0], 1, &pushes(70.0), pushed_at);
    let pushed = published(&monitor, &channel);
    assert_eq!(stated(&pushed), (2, 2), "the file counts the rows it holds");
    assert_eq!(read(&pushed, pushed_at), None, "two rows of three ranks");

    let group = WorkerGroupKey {
        model_id: MODEL.into(),
        worker_type: WorkerType::Regular,
        connection_mode: ConnectionMode::Http,
    };
    poll_group_once_for_test(&monitor, &group, INTERVAL, tick).await;
    let polled = Arc::clone(&channel.borrow());
    assert_eq!(stated(&polled), (3, 3), "the poll's answer, whole");
    assert_eq!(read(&polled, tick), Some(21.0), "the three ranks' sum");

    let next = tick + Duration::from_millis(1);
    monitor.apply_pushed_load(&list[0], 0, &pushes(60.0), next);
    let pushed = published(&monitor, &channel);
    assert_eq!(
        stated(&pushed),
        (2, 2),
        "the poll added no rank to the file"
    );
    assert_eq!(read(&pushed, next), None, "two rows of three ranks again");
}

/// The two token counts a report offers by name are read with no list of
/// named values in the report: a rule of the two keeps the workers whose
/// report says the request fits, and no worker the snapshot holds no
/// report of. The entry that issues no lease keeps the same workers.
#[test]
fn a_rule_of_the_two_token_counts_keeps_the_workers_the_request_fits() {
    let built = built(&format!(
        r#"{{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{{"kv_tokens_capacity":{{"unit":"tokens"}},
                        "kv_tokens_used":{{"unit":"tokens"}}}},
            "profiles":{{"p":{{"filter":["healthy",{ROOM}],
                               "score":["least-requests"],"pick":"best"}}}}}}"#
    ));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    // A request of 200 tokens fits the second worker alone, and then the
    // third alone. The fourth has no report.
    for (fits, used) in [(1, [900, 100, 950]), (2, [801, 900, 800])] {
        let reports = used.into_iter().enumerate();
        let loads = snapshot(reports.map(|(at, used)| (at, holds(used, 1_000))));
        for seed in 0..16 {
            let mut asks = asks(seed, &members, Some(&loads));
            asks.request.facts.prompt_tokens = 200;
            let decision = decided_at(&built.engine, &asks, Instant::now());
            assert_eq!(first_of(&decision), fits, "seed {seed}");
            let bare = built.engine.decide_without_lease(&asks, Instant::now());
            let (bare, _) = bare.expect("a decision");
            assert_eq!(first_of(&bare), fits, "seed {seed}, no lease");
        }
    }
}

/// A list's cells are made where its reports are converted: once a
/// snapshot, by the first decision under it, and the decisions behind it
/// read what it left. Another snapshot converts again, and its values are
/// the ones read.
#[test]
fn a_lists_cells_are_made_once_a_snapshot() {
    let built = built(&reads("", "best"));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    assert_eq!(built.engine.conversions(members.lent()), 0);
    for (nth, base) in [(1, 10.0), (2, 20.0)] {
        let loads = stepped(4, base);
        for seed in 0..32 {
            let read = read_at(&built, &asks(seed, &members, Some(&loads)), Instant::now());
            assert_eq!(read, stepped_values(base), "seed {seed}");
        }
        assert_eq!(
            built.engine.conversions(members.lent()),
            nth,
            "one conversion a snapshot"
        );
    }
}

/// The count of a list's conversions is one document's. An engine that
/// made no value of a list counts none where another engine, made before
/// it, converted the list's reports under its own document. The count
/// makes no value: a list no engine decided over keeps an empty cell.
#[test]
fn a_lists_conversions_are_counted_under_the_asking_engines_document() {
    let (older, newer) = (built(&reads("", "best")), built(&reads("", "best")));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let (loads, now) = (stepped(4, 10.0), Instant::now());
    let counts = || [&older, &newer].map(|built| built.engine.conversions(members.lent()));
    assert_eq!(counts(), [0, 0], "no engine decided over the list");
    assert!(!members.filled(), "a count makes no value");
    let made = newer.engine.values_made();
    let _ = read_at(&older, &asks(7, &members, Some(&loads)), now);
    assert_eq!(counts(), [1, 0], "the newer engine made no value of it");
    assert_eq!(newer.engine.values_made(), made, "and its count made none");
    let _ = read_at(&newer, &asks(7, &members, Some(&loads)), now);
    assert_eq!(counts(), [1, 1], "each under its own document");
}

/// A decision that draws first reads the drawn workers' own values: the
/// cells are the whole list's, by each worker's place, whichever rows the
/// draw asks about.
#[test]
fn a_draw_reads_the_drawn_workers_own_values() {
    let built = built(&reads("", "power-of-two"));
    let list = pool(32);
    let members = PublishedList::new(Arc::clone(&list));
    let loads = stepped(32, 100.0);
    let mut asked = [false; 32];
    for seed in 0..256 {
        built.read.lock().clear();
        let decision = decided_at(
            &built.engine,
            &asks(seed, &members, Some(&loads)),
            Instant::now(),
        );
        let pick = decision.pick().expect("a pick");
        assert!(
            pick.trace.drew_first(),
            "seed {seed}: no pass over the list"
        );
        let read = built.read.lock();
        assert!((1..=2).contains(&read.len()), "seed {seed}: {read:?}");
        for &(place, value) in read.iter() {
            assert_eq!(value, Some(100.0 + f64::from(place)), "seed {seed}");
            asked[place as usize] = true;
        }
    }
    let reached = asked.iter().filter(|&&asked| asked).count();
    assert!(reached > 16, "the draws reached {reached} of 32 workers");
    assert_eq!(
        built.engine.conversions(members.lent()),
        1,
        "one conversion for every draw"
    );
}

/// A session's pin is followed where its worker's signal says the request
/// fits, by the read of the pin's row alone: the list's cells are in the
/// decision's facts ahead of that read. Where the signal says the worker
/// is full, the pin is not followed.
#[test]
fn a_pins_worker_is_asked_its_signal_ahead_of_the_pass_over_the_list() {
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    document.generation = 1;
    for name in ["kv_tokens_capacity", "kv_tokens_used"] {
        let tokens = SignalSpec {
            unit: "tokens".into(),
            max_age_ms: None,
            across_ranks: AcrossRanks::default(),
        };
        document.signals.insert(name.into(), tokens);
    }
    let profile = document.profiles.get_mut("manual");
    let filters = &mut profile.expect("the preset's profile").filter;
    filters.push(serde_json::from_str(ROOM).expect("a filter"));
    let engine = built_of(&document).engine;
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));

    // The third worker alone has room: the key's first request is placed
    // there, and its dispatch pins the key.
    let mut keyed = asks(51, &members, Some(&room_at(4, 2)));
    keyed.request.facts.session_key = Identity::hashed(51);
    let now = Instant::now();
    let mut first =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &keyed, now).expect("a decision");
    let pinned = first.take(0).expect("the first target");
    assert_eq!(pinned.position, 2);
    first.dispatched(now);
    first.completed(&served(pinned.id), now);
    drop(first);

    // Every worker has room: the pin is followed, with no pass over the list.
    keyed.loads = Some(snapshot((0..4).map(|at| (at, holds(100, 1_000)))));
    for _ in 0..3 {
        let followed = decided_at(&engine, &keyed, Instant::now());
        let pick = followed.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        assert_eq!(first_of(&followed), 2);
        assert!(pick.trace.drew_first(), "the pin's row alone was read");
    }
    // The pin's worker is full by its signal, and the first has room.
    keyed.loads = Some(room_at(4, 0));
    let moved = decided_at(&engine, &keyed, Instant::now());
    assert_eq!(first_of(&moved), 0);
}

/// A profile that reads no signal is lent no cells under a document
/// another profile of which reads one: its request is decided as its maker
/// made it. Its decision still converts its list's reports, and the cells
/// that conversion made are the ones a decision of the profile that reads
/// the signal is lent behind it. One list here serves both models: the
/// list's own word is of its first request's model, and the other model is
/// answered by the document, whichever of the two comes first.
#[test]
fn a_profile_that_reads_no_signal_is_lent_no_cells() {
    let built = built(&format!(
        r#"{{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{{"held":{{"unit":"tokens"}}}},
            "profiles":{{"plain":{{"score":["sees-cells"],"pick":"best"}},
                         "reads":{{"score":["reads-held"],"pick":"best"}}}},
            "bindings":[{{"match":{{"model":"{PLAIN_MODEL}"}},"profile":"plain"}},
                        {{"match":{{"model":"*"}},"profile":"reads"}}]}}"#
    ));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let loads = stepped(4, 10.0);
    let mut plain = asks(7, &members, Some(&loads));
    plain.request.facts.model = PLAIN_MODEL;
    for _ in 0..2 {
        let _ = decided_at(&built.engine, &plain, Instant::now());
    }
    assert_eq!(*built.seen.lock(), [false; 2], "no cells in its facts");
    assert!(built.read.lock().is_empty());
    assert_eq!(built.engine.conversions(members.lent()), 1);

    for _ in 0..2 {
        let read = read_at(&built, &asks(7, &members, Some(&loads)), Instant::now());
        assert_eq!(read, stepped_values(10.0));
    }
    assert_eq!(
        built.engine.conversions(members.lent()),
        1,
        "the cells the first decision left"
    );

    // In the other order over another value of the list, the list's word
    // is of the model that reads the signal: the other model's request is
    // still lent no cells.
    let members = PublishedList::new(Arc::clone(&list));
    let read = read_at(&built, &asks(7, &members, Some(&loads)), Instant::now());
    assert_eq!(read, stepped_values(10.0));
    let mut plain = asks(7, &members, Some(&loads));
    plain.request.facts.model = PLAIN_MODEL;
    built.seen.lock().clear();
    for _ in 0..2 {
        let _ = decided_at(&built.engine, &plain, Instant::now());
    }
    assert_eq!(*built.seen.lock(), [false; 2], "no cells in its facts");
}

/// The stock scorer of costs reads a named signal as one of its terms
/// (`signal.<name>`): the worker whose report states the least is named
/// first, wherever it stands in the list.
#[test]
fn a_cost_term_of_a_named_signal_names_the_worker_that_states_the_least() {
    let built = built(
        r#"{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{"held":{"unit":"tokens"}},
            "profiles":{"p":{"score":[{"least-cost":{"terms":{"signal.held":1},
                                                      "unreported":"missing-as-zero"}}],
                             "pick":"best"}}}"#,
    );
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    for least in 0..4 {
        let value = |at| if at == least { 10.0 } else { 500.0 };
        let loads = snapshot((0..4).map(|at| (at, report(held(value(at))))));
        for seed in 0..8 {
            let asks = asks(seed, &members, Some(&loads));
            let decision = decided_at(&built.engine, &asks, Instant::now());
            assert_eq!(first_of(&decision) as usize, least, "seed {seed}");
        }
    }
}

/// The entry that issues no lease is lent the list's cells as the leasing
/// entry is: the cells are lent where a made single request is decided,
/// and both entries decide there.
#[test]
fn the_entry_that_issues_no_lease_is_lent_the_cells_too() {
    let built = built(&reads("", "best"));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let loads = stepped(4, 10.0);
    let asks = asks(7, &members, Some(&loads));
    let decided = built.engine.decide_without_lease(&asks, Instant::now());
    assert!(decided.is_ok(), "a decision");
    let mut read = mem::take(&mut *built.read.lock());
    read.sort_by_key(|&(place, _)| place);
    assert_eq!(read, stepped_values(10.0));
}

/// A binding's shadow profile decides over the facts its live decision was
/// lent. Here the live profile reads no signal and its shadow reads one:
/// the shadow's stage reads each worker's value, and the shadow's decision
/// is counted behind the issue as under any other document.
#[test]
fn a_shadow_profile_reads_the_cells_its_live_decision_was_lent() {
    let document = r#"{"apiVersion":"smg.policy/v1","generation":1,
        "signals":{"held":{"unit":"tokens"}},
        "profiles":{"plain":{"score":["sees-cells"],"pick":"best"},
                    "reads":{"score":["reads-held"],"pick":"best"}},
        "bindings":[{"match":{"model":"*"},"profile":"plain"}]}"#;
    let mut document: PolicySpec = serde_json::from_str(document).expect("a document");
    document.bindings[0].shadow = Some(ShadowSpec::Profile("reads".into()));
    let built = built_of(&document);
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let loads = stepped(4, 10.0);
    let read = read_at(&built, &asks(7, &members, Some(&loads)), Instant::now());
    assert_eq!(read, stepped_values(10.0), "what the shadow's stage read");
    let counted = ShadowKind::ALL.map(|kind| built.engine.shadowed(0, kind));
    assert_eq!(counted.iter().sum::<u64>(), 1, "{counted:?}");
}

/// A lease is issued at its decision's instant whether the decision was
/// lent cells or not. A request that is lent cells is decided as a copy
/// that carries them, and its entry issues by the request itself: what the
/// decision wrote, its instant, is handed back to the request.
#[test]
fn a_lease_is_issued_at_its_decisions_instant_where_cells_were_lent() {
    let built = built(&format!(
        r#"{{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{{"held":{{"unit":"tokens"}}}},
            "profiles":{{"plain":{{"score":["hears-issue"],"pick":"best"}},
                         "reads":{{"score":["reads-held","hears-issue"],"pick":"best"}}}},
            "bindings":[{{"match":{{"model":"{PLAIN_MODEL}"}},"profile":"plain"}},
                        {{"match":{{"model":"*"}},"profile":"reads"}}]}}"#
    ));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let loads = stepped(4, 10.0);
    let now = Instant::now() + Duration::from_secs(1);
    let mut plain = asks(7, &members, Some(&loads));
    plain.request.facts.model = PLAIN_MODEL;
    let _ = decided_at(&built.engine, &plain, now);
    let plain = mem::take(&mut *built.issued.lock());
    assert!(built.read.lock().is_empty(), "no signal was read");
    assert!(!plain.is_empty() && !plain.contains(&0), "{plain:?}");

    let _ = decided_at(&built.engine, &asks(7, &members, Some(&loads)), now);
    assert_eq!(built.read.lock().len(), 4, "the signal was read");
    assert_eq!(*built.issued.lock(), plain, "the same instant");
}

/// Each rank of a data-parallel worker that is a row of its own reads its
/// own values, each ended by its own sample. The two rows here are the two
/// ranks of one worker, and each pushes its records through the worker
/// crate's own monitor, whose report of a row holds that rank alone. Both
/// state a value. Then the second rank stops reporting: two seconds on,
/// under a maximum age of one, the first rank's new value is read and the
/// second's has ended.
#[tokio::test]
async fn each_rank_that_is_a_row_reads_its_own_pushed_values() {
    let built = built(&reads(r#","max_age_ms":1000"#, "best"));
    let list: Arc<[Arc<dyn Worker>]> = [0, 1].map(rank_row).into();
    let members = PublishedList::new(Arc::clone(&list));
    let (monitor, channel) = monitor_over(&list);
    let start = Instant::now();
    monitor.apply_pushed_load(&list[0], 0, &pushes(5.0), start);
    monitor.apply_pushed_load(&list[1], 1, &pushes(7.0), start);
    let loads = published(&monitor, &channel);
    let ranks = |report: &WorkerLoadResponse| report.loads.len();
    let both = [0, 1].map(|at| loads.get(list[at].url()).map(ranks));
    assert_eq!(both, [Some(1); 2], "each row's report is its own rank's");
    let read = read_at(&built, &asks(7, &members, Some(&loads)), start);
    assert_eq!(read, [(0, Some(5.0)), (1, Some(7.0))]);

    let later = start + Duration::from_secs(2);
    monitor.apply_pushed_load(&list[0], 0, &pushes(6.0), later);
    let loads = published(&monitor, &channel);
    let read = read_at(&built, &asks(7, &members, Some(&loads)), later);
    assert_eq!(read, [(0, Some(6.0)), (1, None)]);
}

/// A worker that pushes its load is read its two token counts between the
/// records that state them. A record of an event batch states neither: the
/// worker crate's monitor keeps both as they were last stated, and its
/// report offers each by name as old as that statement. So the rule of
/// the two counts still keeps the worker a request fits behind such a
/// record. Each count ends by its signal's maximum age, counted from the
/// record that stated it: past it no worker reads a value, none has room,
/// and the rule, which is soft, lets every worker stand.
#[tokio::test]
async fn a_pushing_workers_token_counts_are_read_behind_a_record_that_leaves_them_out() {
    let built = built(&format!(
        r#"{{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{{"kv_tokens_capacity":{{"unit":"tokens","max_age_ms":500}},
                        "kv_tokens_used":{{"unit":"tokens","max_age_ms":500}}}},
            "profiles":{{"p":{{"filter":["healthy",{ROOM}],
                               "score":["least-requests"],"pick":"best"}}}}}}"#
    ));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let (monitor, channel) = monitor_over(&list);
    let start = Instant::now();
    // A request of 200 tokens fits the second worker alone.
    for (worker, used) in list.iter().zip([900, 100]) {
        let states = common_proto::EngineLoad {
            num_used_tokens: Some(used),
            max_total_num_tokens: Some(1_000),
            ..Default::default()
        };
        monitor.apply_pushed_load(worker, 0, &states, start);
    }
    // 200 ms on each sends a record that is the core alone.
    let core = common_proto::EngineLoad::default();
    for worker in list.iter() {
        monitor.apply_pushed_load(worker, 0, &core, start + Duration::from_millis(200));
    }
    let loads = published(&monitor, &channel);
    let (report, sampled) = loads.report(list[0].url()).expect("a report");
    let counts = (
        report.loads[0].num_used_tokens,
        report.loads[0].max_total_num_tokens,
    );
    assert_eq!(counts, (900, 1_000), "the entry keeps both figures");
    let picked = |seed: u64, after: u64| {
        let mut asks = asks(seed, &members, Some(&loads));
        asks.request.facts.prompt_tokens = 200;
        let now = sampled + Duration::from_millis(after);
        first_of(&decided_at(&built.engine, &asks, now))
    };
    // The counts are 200 ms old at the record's sample, and end 500 ms
    // behind the record that stated them. Read, they leave the second
    // worker alone with room. Ended, no worker has room, every worker
    // stands, and the first is picked too.
    for seed in 0..16 {
        assert_eq!(picked(seed, 300), 1, "seed {seed}");
    }
    assert!((0..16).any(|seed| picked(seed, 301) == 0));
}

/// A worker object that stands for several ranks and pushes its load gives
/// one value of its ranks', each rank read at its own sample. Its report
/// lists each rank at that rank's latest record, and each entry states its
/// own sample instant. The first worker here stands for two ranks and
/// pushes over both, and the second pushes over its one, through the
/// worker crate's own monitor, whose report
/// is the one read: the first reads the larger of its two ranks' values.
/// Then the first worker's second rank stops reporting. Two seconds on,
/// under a maximum age of one second, that rank's last value has ended and
/// the worker's value with it. Read under the report's one instant, the
/// newest record's, the stopped rank's value would stand as new.
#[tokio::test]
async fn a_worker_that_pushes_over_several_ranks_gives_one_value_until_a_rank_stops() {
    let built = built(&reads(r#","max_age_ms":1000,"across_ranks":"max""#, "best"));
    let list: Arc<[Arc<dyn Worker>]> = [wide(0, 2), ready_worker(&url(1), MODEL)].into();
    let members = PublishedList::new(Arc::clone(&list));
    let (monitor, channel) = monitor_over(&list);
    let start = Instant::now();
    monitor.apply_pushed_load(&list[0], 0, &pushes(5.0), start);
    monitor.apply_pushed_load(&list[0], 1, &pushes(7.0), start);
    monitor.apply_pushed_load(&list[1], 0, &pushes(9.0), start);
    let loads = published(&monitor, &channel);
    let ranks = |report: &WorkerLoadResponse| report.loads.len();
    let listed = [0, 1].map(|at| loads.get(&url(at)).map(ranks));
    assert_eq!(listed, [Some(2), Some(1)]);
    let read = read_at(&built, &asks(7, &members, Some(&loads)), start);
    assert_eq!(read, [(0, Some(7.0)), (1, Some(9.0))]);

    let later = start + Duration::from_secs(2);
    monitor.apply_pushed_load(&list[0], 0, &pushes(6.0), later);
    monitor.apply_pushed_load(&list[1], 0, &pushes(9.5), later);
    let loads = published(&monitor, &channel);
    let read = read_at(&built, &asks(7, &members, Some(&loads)), later);
    assert_eq!(read, [(0, None), (1, Some(9.5))]);
}

/// A request for runtime 7 gets a pick.
/// The scorer reads 11 tokens from the second worker and 13 from the fourth.
/// It reads no other worker.
#[test]
fn rows_a_demand_leaves_read_their_own_signal_values() {
    use crate::engine::{DemandFacts, FixedFacts};

    let built = built(&reads("", "best"));
    let list = pool(4);
    let mut runtimes = [9, 7, 9, 7].into_iter();
    let members = PublishedList::with_facts(Arc::clone(&list), |_| FixedFacts {
        demands: DemandFacts {
            runtime: runtimes.next().expect("one runtime per worker"),
            ..DemandFacts::default()
        },
        ..FixedFacts::default()
    });
    let loads = stepped(4, 10.0);
    let mut asked = asks(7, &members, Some(&loads));
    asked.request.constraints.demands.runtime = 7;

    let decision = decided_at(&built.engine, &asked, Instant::now());
    assert!(decision.pick().is_some(), "{decision:?}");
    let mut read = mem::take(&mut *built.read.lock());
    read.sort_by_key(|&(place, _)| place);
    assert_eq!(read, [(1, Some(11.0)), (3, Some(13.0))]);
}

/// A request that stands alone is asked its list's signal word at its own
/// phase. Here the binding of the prefill phase names the profile that
/// reads `held`, and the binding behind it names one that reads no signal.
/// A prefill over one list reads each worker's stated value at both
/// entries, and a single request over a list of its own reads none.
#[test]
fn a_request_that_stands_alone_reads_the_signal_of_its_own_phases_binding() {
    use crate::engine::view::Phase;

    let document = r#"{"apiVersion":"smg.policy/v1","generation":1,
        "signals":{"held":{"unit":"tokens"}},
        "profiles":{"p":{"score":["reads-held"],"pick":"best"},
                    "q":{"pick":"round-robin"}},
        "bindings":[
            {"match":{"model":"*","phase":["prefill"]},"profile":"p"},
            {"match":{"model":"*","phase":[]},"profile":"q"}]}"#;
    let built = built(document);
    let list = pool(4);
    let loads = stepped(4, 10.0);
    let prefills = PublishedList::new(Arc::clone(&list));
    let mut alone = asks(7, &prefills, Some(&loads));
    alone.request.facts.phase = Phase::Prefill;
    let now = Instant::now();
    assert_eq!(read_at(&built, &alone, now), stepped_values(10.0));
    // The entry that issues no lease asks the same word.
    built.read.lock().clear();
    let bare = built.engine.decide_without_lease(&alone, now);
    assert!(bare.expect("a decision").0.pick().is_some());
    let mut read = mem::take(&mut *built.read.lock());
    read.sort_by_key(|&(place, _)| place);
    assert_eq!(read, stepped_values(10.0));
    // A single request resolves to the binding behind, whose profile reads
    // no signal: the stage that reads one is never asked.
    let singles = PublishedList::new(Arc::clone(&list));
    let single = asks(7, &singles, Some(&loads));
    assert_eq!(read_at(&built, &single, now), []);
}
