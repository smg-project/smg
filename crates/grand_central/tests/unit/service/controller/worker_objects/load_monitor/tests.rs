use std::{
    collections::HashMap,
    sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering::SeqCst},
};

use async_trait::async_trait;
use smg_worker::{
    common_proto::KvEventBatch,
    connection::engine::{AdminAnswer, EngineConnection, EngineConnector, HealthAnswer, ZmqLink},
    health::overload::OverloadThresholds,
    worker::{ProfileOptions, RuntimeType, SchedulerLoadSnapshot, WorkerLoadResponse},
};
use tokio::{task::JoinSet, time};

use super::*;
use crate::{
    engine::view::WireModes,
    service::controller::{frame_loads::FrameLoads, Config},
    store::{GatewayId, MemoryStore, Reachability, Registrant, WorkerFacts, WorkersStore},
};

const A: &str = "grpc://a:9000";
const B: &str = "grpc://b:9000";
const C: &str = "grpc://c:9000";

/// How often the replica's monitor polls a worker.
const EVERY: Duration = Duration::from_secs(2);

/// An engine that answers a load poll with its queue, as late as the
/// case says, and counts the polls. The worker crate's scripted engine
/// answers a load poll in its own tests alone.
#[derive(Debug, Default)]
struct Engine {
    polls: AtomicUsize,
    waiting: AtomicI32,
    /// How long a poll's answer takes, in milliseconds.
    late: AtomicU64,
    /// What a case does where a poll in flight is dropped unanswered,
    /// as the monitor's stop drops it.
    at_cut: parking_lot::Mutex<Option<AtCut>>,
}

/// A case's step for a poll that is dropped unanswered.
struct AtCut(Box<dyn Fn() + Send + Sync>);

impl std::fmt::Debug for AtCut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AtCut")
    }
}

/// A poll in flight: dropped unanswered, it runs the engine's step.
struct InFlight<'a> {
    engine: &'a Engine,
    answered: bool,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if self.answered {
            return;
        }
        if let Some(step) = self.engine.at_cut.lock().as_ref() {
            (step.0)();
        }
    }
}

fn not_scripted<T>() -> Result<T, tonic::Status> {
    Err(tonic::Status::unimplemented(
        "the case scripts no such call",
    ))
}

#[async_trait]
impl EngineConnection for Engine {
    fn carries_admin_calls(&self) -> bool {
        false
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        not_scripted()
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        self.polls.fetch_add(1, SeqCst);
        let mut poll = InFlight {
            engine: self,
            answered: false,
        };
        time::sleep(Duration::from_millis(self.late.load(SeqCst))).await;
        poll.answered = true;
        Ok(WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_reqs: self.waiting.load(SeqCst),
                ..Default::default()
            }],
            ..Default::default()
        })
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        not_scripted()
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        not_scripted()
    }

    async fn subscribe_kv_events(
        &self,
        _start_seq: u64,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        not_scripted()
    }

    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        not_scripted()
    }
}

/// A connector whose every gRPC connect is answered by the one engine.
#[derive(Debug)]
struct Dials(Arc<Engine>);

#[async_trait]
impl EngineConnector for Dials {
    async fn connect_grpc(
        &self,
        _base_url: &str,
        _runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        Ok(self.0.clone())
    }

    async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        Err("the case scripts no such connect".to_owned())
    }

    async fn dial_grpc(&self, _base_url: &str, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

/// A row of a gRPC worker under `key`.
fn facts(key: &str) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation: 1,
        models: vec!["m".to_owned()],
        dp_size: 1,
        wire_modes: WireModes::GRPC,
        ..WorkerFacts::default()
    }
}

/// [`facts`] of a worker that serves `model` alone: the monitor polls
/// each model's workers as a group of their own.
fn serving(key: &str, model: &str) -> WorkerFacts {
    WorkerFacts {
        models: vec![model.to_owned()],
        ..facts(key)
    }
}

/// A replica: its store, its worker objects over one engine the case
/// scripts, the monitor's one channel, the role's word and the
/// monitor's task. No socket: the engine answers in process.
struct Replica {
    store: Arc<MemoryStore>,
    workers: Arc<WorkerObjects>,
    engine: Arc<Engine>,
    /// The monitor's channel, taken once ahead of the task's start.
    loads: LoadReceiver,
    /// The reader of the monitor's counts, taken as the channel is.
    counts: Box<dyn Fn() -> LoadTotals>,
    /// The sink of the streams' load records, taken as the channel is.
    pushed: Arc<PushedRecords>,
    role: watch::Sender<Option<u64>>,
    stop: watch::Sender<bool>,
    task: JoinSet<()>,
}

impl Replica {
    /// A replica that does not lead yet, its monitor's task started.
    fn new() -> Self {
        Self::stating(Config::DEFAULT_WORKER_OVERLOAD)
    }

    /// [`Self::new`] of a start that states `overload`.
    fn stating(overload: OverloadThresholds) -> Self {
        let store = Arc::new(MemoryStore::new());
        let engine = Arc::new(Engine::default());
        let dials: Arc<dyn EngineConnector> = Arc::new(Dials(Arc::clone(&engine)));
        let workers = Arc::new(WorkerObjects::with_connector(
            Arc::clone(&store),
            dials,
            overload,
        ));
        let monitor = LoadMonitor::new(&workers, EVERY);
        let loads = monitor.loads();
        let counts = Box::new(monitor.counts());
        let pushed = Arc::clone(monitor.pushed());
        let (role, led) = watch::channel(None);
        let (stop, stopped) = watch::channel(false);
        let mut task = JoinSet::new();
        task.spawn(async move { monitor.run(stopped, led).await });
        Self {
            store,
            workers,
            engine,
            loads,
            counts,
            pushed,
            role,
            stop,
            task,
        }
    }

    /// Writes the row of `facts` through the holder and readies its
    /// object by the connect signal, as a registration call does.
    fn register(&self, facts: WorkerFacts) {
        let holder = self.workers.holder();
        let written = holder.upsert(&self.store, facts, Registrant::Control);
        let row = written.expect("the store takes the write").id();
        let standing = self.store.get(row).expect("the row");
        holder.connect_row(&self.store, row, standing.revision);
    }

    /// How many load polls the engine answered.
    fn polls(&self) -> usize {
        self.engine.polls.load(SeqCst)
    }

    /// The queue the monitor's snapshot states for the worker at
    /// `url`: none where it holds no report of that worker.
    fn waiting(&self, url: &str) -> Option<i64> {
        let snapshot = self.loads.borrow();
        snapshot
            .get(url)
            .map(WorkerLoadResponse::total_waiting_reqs)
    }

    /// The ranks the monitor's snapshot states for the worker at `url`,
    /// as its report lists them: none where it holds no report.
    fn ranks(&self, url: &str) -> Vec<i32> {
        let snapshot = self.loads.borrow();
        let ranks = snapshot.get(url).map(|report| &report.loads);
        ranks
            .into_iter()
            .flatten()
            .map(|rank| rank.dp_rank)
            .collect()
    }

    /// Whether an overload verdict stands on the object at `url`.
    fn overloaded(&self, url: &str) -> bool {
        let registry = self.workers.objects().registry();
        let object = registry.get_by_url(url).expect("the row's object");
        object.is_overloaded()
    }

    /// Hands the sink a load record of the worker at `url`, as the task
    /// of the worker's event stream does: `waiting` requests queue.
    fn push(&self, url: &str, waiting: u32) {
        self.push_rank(url, 0, waiting);
    }

    /// The same record for `rank` of the worker's engine.
    fn push_rank(&self, url: &str, rank: i32, waiting: u32) {
        let registry = self.workers.objects().registry();
        let object = registry.get_by_url(url).expect("the row's object");
        let record = EngineLoad {
            waiting_requests: waiting,
            ..EngineLoad::default()
        };
        Arc::clone(&self.pushed).push_load(&object, rank, &record, Instant::now());
    }

    /// Whether the object at `url` is marked as a worker whose stream
    /// pushes its load: the monitor marks it at the first record it
    /// takes.
    fn pushes(&self, url: &str) -> bool {
        let registry = self.workers.objects().registry();
        let object = registry.get_by_url(url).expect("the row's object");
        let state = object.runtime_state();
        state.is_some_and(|state| state.pushes_load_records())
    }

    /// Tells the stop: whether the monitor's task ended clean.
    async fn stopped(&mut self) -> bool {
        self.stop.send_replace(true);
        matches!(self.task.join_next().await, Some(Ok(())))
    }
}

/// Lets the replica's tasks run until `holds`, a millisecond of the
/// paused clock at a time and well inside one poll interval: whether it
/// came to hold.
async fn comes(mut holds: impl FnMut() -> bool) -> bool {
    for _ in 0..200 {
        if holds() {
            return true;
        }
        time::sleep(Duration::from_millis(1)).await;
    }
    holds()
}

/// The monitor judges a worker whose row names no threshold by the two
/// its start states. Five wait: a replica started with a count of four
/// vetoes the worker at its first poll, and one started with the
/// default of eight does not.
#[tokio::test(start_paused = true)]
async fn a_worker_that_names_no_threshold_is_judged_by_its_starts() {
    let four = OverloadThresholds {
        waiting_requests: Some(4),
        ..Config::DEFAULT_WORKER_OVERLOAD
    };
    for (overload, vetoed) in [(four, true), (Config::DEFAULT_WORKER_OVERLOAD, false)] {
        let mut replica = Replica::stating(overload);
        replica.register(facts(A));
        replica.engine.waiting.store(5, SeqCst);
        replica.role.send_replace(Some(1));
        assert!(comes(|| replica.waiting(A) == Some(5)).await, "the report");
        // The verdict is the first poll's own: the next is an interval
        // away, and a moment more moves nothing.
        time::sleep(Duration::from_millis(5)).await;
        assert_eq!(replica.overloaded(A), vetoed, "{overload:?}");
        assert!(replica.stopped().await, "the task ends at the stop");
    }
}

/// The reader of the monitor's counts answers what the replica's polls
/// did and how often an overload verdict of a worker changed: nothing
/// before a lead, then the lead's poll, its row and the verdict it
/// wrote. The lead's end takes the verdict off, a flip more, and the
/// counts stand through it: they are the replica's, not one lead's.
#[tokio::test(start_paused = true)]
async fn the_counts_read_a_leads_polls_and_each_verdict_that_changed() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    assert_eq!((replica.counts)(), LoadTotals::default(), "no lead yet");

    // Nine wait: over the eight of the program's default.
    replica.engine.waiting.store(9, SeqCst);
    replica.role.send_replace(Some(1));
    assert!(comes(|| replica.overloaded(A)).await);
    let led = (replica.counts)();
    assert_eq!((led.polls.polls, led.polls.rows), (1, 1));
    assert_eq!(led.overload_flips, 1, "the poll's verdict");

    replica.role.send_replace(None);
    assert!(comes(|| !replica.overloaded(A)).await);
    let behind = (replica.counts)();
    assert_eq!(behind.polls, led.polls, "the counts stand through the stop");
    assert_eq!(behind.overload_flips, 2, "the lead's end took it off");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A replica that does not lead polls no engine for its load.
#[tokio::test(start_paused = true)]
async fn a_standby_polls_nothing() {
    let mut replica = Replica::new();
    replica.register(facts(A));

    time::sleep(EVERY * 3).await;
    assert_eq!(replica.polls(), 0, "a standby polled");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A load record that comes while the replica leads none is dropped: no
/// report of it is published, no verdict is written, and its worker is
/// not marked as one whose stream pushes. The monitor is handed
/// nothing, so it starts no task for the record.
#[tokio::test(start_paused = true)]
async fn a_standby_drops_a_pushed_load_record() {
    let mut replica = Replica::new();
    replica.register(facts(A));

    // Nine wait: over the eight of the program's default.
    replica.push(A, 9);
    time::sleep(EVERY).await;
    assert_eq!(replica.waiting(A), None, "a standby published a report");
    assert!(!replica.overloaded(A), "a standby wrote a verdict");
    assert!(!replica.pushes(A), "a standby marked the worker");
    assert_eq!(replica.polls(), 0, "a standby polled");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A load record that comes while the replica leads reaches the monitor
/// as a poll does: its report is published and judged at once, and its
/// worker, whose records are fresh, is left to its stream at the next
/// tick.
#[tokio::test(start_paused = true)]
async fn a_leader_takes_a_pushed_load_record_as_a_poll() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    replica.role.send_replace(Some(1));
    assert!(
        comes(|| replica.waiting(A) == Some(0)).await,
        "the take's poll"
    );

    // Nine wait: over the eight of the program's default. The engine
    // still answers a poll with none.
    replica.push(A, 9);
    assert!(replica.overloaded(A), "the record is judged at once");
    assert!(replica.pushes(A), "the monitor took the record");
    assert!(
        comes(|| replica.waiting(A) == Some(9)).await,
        "the record's report is published"
    );

    let (polled, left) = (replica.polls(), (replica.counts)().polls.suppressed);
    time::sleep(EVERY).await;
    assert!(
        comes(|| (replica.counts)().polls.suppressed == left + 1).await,
        "the tick left the worker to its stream"
    );
    assert_eq!(
        replica.polls(),
        polled,
        "a worker of fresh records was polled"
    );
    assert_eq!(replica.waiting(A), Some(9), "the record's report stands");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A load record that comes behind the lead's end lands behind no stop:
/// the sink drops it once the word is turned, at whichever turn of the
/// stopping task it comes, and a lead that ended keeps the streams shut
/// out.
#[tokio::test(start_paused = true)]
async fn a_pushed_load_record_lands_behind_no_stop() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    replica.role.send_replace(Some(1));
    assert!(
        comes(|| replica.waiting(A) == Some(0)).await,
        "the take's poll"
    );

    // The word turns, and a record comes at every turn the stopping
    // task yields: nine wait, over the eight of the program's default.
    replica.role.send_replace(None);
    for _ in 0..64 {
        replica.push(A, 9);
        tokio::task::yield_now().await;
    }
    time::sleep(EVERY).await;
    assert_eq!(replica.waiting(A), None, "a record landed behind the stop");
    assert!(!replica.overloaded(A), "a verdict stands behind the stop");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A load record that comes while the stop waits for the monitor's
/// tasks is dropped: the streams are shut out ahead of the stop. The
/// record comes where the stop drops a poll in flight. Handed on there,
/// it would start a publishing task that the stop no longer ends and
/// mark its worker as one whose stream pushes. The stop takes such a
/// record's report and its mark, and not the task: the mark is read
/// where the record came.
#[tokio::test(start_paused = true)]
async fn a_load_record_inside_the_stop_is_dropped() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    replica.role.send_replace(Some(1));
    assert!(
        comes(|| replica.waiting(A) == Some(0)).await,
        "the take's poll"
    );

    // The period's poll is answered late, and the lead ends while it
    // is in flight: nine wait in the record that comes at its drop,
    // over the eight of the program's default.
    replica.engine.late.store(60_000, SeqCst);
    let polled = replica.polls();
    time::sleep(EVERY).await;
    assert!(
        comes(|| replica.polls() == polled + 1).await,
        "the period's poll is in flight"
    );
    let pushed = Arc::clone(&replica.pushed);
    let registry = replica.workers.objects().registry();
    let object = registry.get_by_url(A).expect("the row's object");
    // Whether the object was marked where the record came.
    let marked = Arc::new(parking_lot::Mutex::new(None));
    let at_record = Arc::clone(&marked);
    *replica.engine.at_cut.lock() = Some(AtCut(Box::new(move || {
        let record = EngineLoad {
            waiting_requests: 9,
            ..EngineLoad::default()
        };
        Arc::clone(&pushed).push_load(&object, 0, &record, Instant::now());
        let state = object.runtime_state();
        *at_record.lock() = Some(state.is_some_and(|state| state.pushes_load_records()));
    })));
    replica.role.send_replace(None);
    time::sleep(EVERY).await;
    assert_eq!(replica.waiting(A), None, "a record landed inside the stop");
    assert!(!replica.overloaded(A), "a verdict stands behind the stop");
    assert_eq!(
        *marked.lock(),
        Some(false),
        "the monitor took a record inside the stop"
    );
    // The step holds the sink, and the engine holds the step.
    replica.engine.at_cut.lock().take();
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// Nothing that a stream filed under one lead is published under a later
/// one. Under the first lead B's rank 1 pushed and was published, and
/// A's record came inside the lead's last publish window: it is still
/// queued where the lead ends. The monitor's stop evicts both files,
/// and takes the mark of a stream that pushes off both objects.
/// The next lead's first window, which B's rank 0 opens while no poll
/// has answered yet, publishes that record alone: no rank of B's from
/// the earlier lead, and no report of A. C never pushes: its poll goes
/// out at each take, whatever the monitor keeps of the other two.
#[tokio::test(start_paused = true)]
async fn a_report_filed_under_one_lead_is_published_under_no_later_one() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    replica.register(facts(B));
    replica.register(facts(C));
    replica.role.send_replace(Some(1));
    assert!(
        comes(|| replica.waiting(A) == Some(0) && replica.waiting(B) == Some(0)).await,
        "the take's poll"
    );

    // Five wait on B's rank 1, and its window publishes them under the
    // rank the record named.
    replica.push_rank(B, 1, 5);
    assert!(
        comes(|| replica.waiting(B) == Some(5)).await,
        "the window publishes B's rank 1"
    );
    assert_eq!(replica.ranks(B), vec![1], "the record's rank");
    // A's record, and the lead ends inside its publish window.
    replica.push(A, 9);
    replica.role.send_replace(None);
    assert!(
        comes(|| replica.waiting(A).is_none() && replica.waiting(B).is_none()).await,
        "the lead's end empties the snapshot"
    );
    assert!(
        !replica.pushes(A) && !replica.pushes(B),
        "an object is marked behind the lead's end"
    );

    // The next lead's polls are answered late, so none lays over a
    // report, and B's rank 0 opens that lead's first publish window.
    replica.engine.late.store(60_000, SeqCst);
    let polled = replica.polls();
    replica.role.send_replace(Some(2));
    assert!(
        comes(|| replica.polls() > polled).await,
        "the take's poll is in flight"
    );
    replica.push(B, 1);
    assert!(
        comes(|| replica.waiting(B).is_some()).await,
        "the window publishes B's record"
    );
    assert_eq!(
        replica.waiting(B),
        Some(1),
        "a rank of the earlier lead is published"
    );
    assert_eq!(
        replica.waiting(A),
        None,
        "a report of the earlier lead is published"
    );
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A leader polls each `Ready` worker and publishes their reports: one
/// that stood before its term at once, and one that comes during it at
/// once too, by one poll of its own. Its group's next poll still comes
/// on its beat, an interval behind the term's first, and takes both.
#[tokio::test(start_paused = true)]
async fn a_leader_polls_its_workers_and_publishes_their_reports() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    replica.engine.waiting.store(3, SeqCst);
    replica.role.send_replace(Some(1));
    assert!(
        comes(|| replica.waiting(A) == Some(3)).await,
        "the term's first poll is published"
    );

    time::sleep(EVERY / 2).await;
    let polled = replica.polls();
    replica.engine.waiting.store(5, SeqCst);
    replica.register(facts(B));
    assert!(
        comes(|| replica.waiting(B) == Some(5)).await,
        "a worker that comes during the term is asked at once"
    );
    time::sleep(EVERY / 2 - Duration::from_millis(100)).await;
    assert_eq!(
        replica.polls(),
        polled + 1,
        "the worker that came is asked once inside the period, and no other"
    );
    assert_eq!(
        replica.waiting(A),
        Some(3),
        "the report of the worker that stood is left to the period's poll"
    );

    replica.engine.waiting.store(7, SeqCst);
    assert!(
        comes(|| replica.waiting(A) == Some(7) && replica.waiting(B) == Some(7)).await,
        "the period's poll comes on its beat and takes both workers"
    );
    assert_eq!(replica.polls(), polled + 3, "one poll a worker on the beat");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A row that one gateway alone reaches is asked for no load: the
/// replica cannot dial it. Its object is `Ready` and stands in the
/// monitor's group, and no report of it is published.
#[tokio::test(start_paused = true)]
async fn a_row_one_gateway_alone_reaches_is_asked_for_no_load() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    replica.register(WorkerFacts {
        reachable_from: Reachability::Gateway(GatewayId(1)),
        ..facts(B)
    });
    replica.role.send_replace(Some(1));
    assert!(comes(|| replica.waiting(A) == Some(0)).await);

    time::sleep(EVERY * 3).await;
    assert_eq!(
        replica.waiting(B),
        None,
        "the replica asked a row it cannot dial"
    );
    assert_eq!(replica.polls(), 4, "one poll a period, of the one worker");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// The word that nobody leads here stops the monitor: no poll goes out,
/// the overload verdict it wrote is off its worker, and its snapshot is
/// empty. The next lead starts the same monitor again: it polls at
/// once and publishes on the one channel. A word that goes from one
/// term to the next leaves the monitor running.
#[tokio::test(start_paused = true)]
async fn a_leads_end_stops_the_monitor_and_the_next_lead_starts_it_again() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    // Nine wait: over the eight of the program's default.
    replica.engine.waiting.store(9, SeqCst);
    replica.role.send_replace(Some(1));
    assert!(comes(|| replica.waiting(A) == Some(9)).await);
    assert!(replica.overloaded(A), "judged by the default thresholds");

    replica.role.send_replace(None);
    assert!(
        comes(|| !replica.overloaded(A)).await,
        "the verdict is taken off at the lead's end"
    );
    assert_eq!(replica.waiting(A), None, "the snapshot is published empty");
    let polled = replica.polls();
    time::sleep(EVERY * 3).await;
    assert_eq!(replica.polls(), polled, "a poll went out after the lead");

    replica.engine.waiting.store(1, SeqCst);
    replica.role.send_replace(Some(2));
    assert!(
        comes(|| replica.waiting(A) == Some(1)).await,
        "the next lead polls at once, on the one channel"
    );
    assert!(!replica.overloaded(A));

    // From one term to the next with no word between: the monitor is
    // not started again, so no poll goes out ahead of its interval.
    let polled = replica.polls();
    replica.role.send_replace(Some(3));
    assert!(
        !comes(|| replica.polls() > polled).await,
        "a term's change started the monitor again"
    );
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// What a deciding thread reads across a lead, its loss, a follow and
/// the next take ([`FrameLoads`]): the lead's report while the replica
/// leads, none from the loss on, and none at the take until the take's
/// own first poll lands. The engine's queue moves during the follow and
/// the take's poll answers late: the report of the lead before is read
/// at no instant of the take.
#[tokio::test(start_paused = true)]
async fn a_take_behind_a_follow_reads_no_report_of_the_lead_before() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    let mut thread = FrameLoads::of(replica.loads.clone());
    let read = |thread: &mut FrameLoads| {
        let report = thread.newest().get(A);
        report.map(WorkerLoadResponse::total_waiting_reqs)
    };
    replica.engine.waiting.store(3, SeqCst);
    replica.role.send_replace(Some(1));
    assert!(comes(|| replica.waiting(A) == Some(3)).await);
    assert_eq!(read(&mut thread), Some(3), "the lead's own report");

    replica.role.send_replace(None);
    assert!(
        comes(|| replica.waiting(A).is_none()).await,
        "the snapshot is published empty at the loss"
    );
    assert_eq!(read(&mut thread), None, "a report was read behind the loss");

    // The follow: the engine's queue moves, and nothing is read of it.
    replica.engine.waiting.store(7, SeqCst);
    time::sleep(EVERY * 3).await;
    assert_eq!(read(&mut thread), None, "a follower read a report");

    // The take, whose first poll answers half a second late.
    let polled = replica.polls();
    replica.engine.late.store(500, SeqCst);
    replica.role.send_replace(Some(2));
    assert!(
        comes(|| replica.polls() == polled + 1).await,
        "the take's poll is out"
    );
    assert_eq!(
        read(&mut thread),
        None,
        "the take read the report of the lead before"
    );
    time::sleep(Duration::from_millis(500)).await;
    assert!(comes(|| replica.waiting(A) == Some(7)).await);
    assert_eq!(read(&mut thread), Some(7), "the take's own first report");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// The stop ends the monitor with its task: its verdicts are taken off,
/// and nothing polls behind it.
#[tokio::test(start_paused = true)]
async fn the_stop_ends_the_monitor() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    // Nine wait: over the eight of the program's default.
    replica.engine.waiting.store(9, SeqCst);
    replica.role.send_replace(Some(1));
    assert!(comes(|| replica.overloaded(A)).await);

    assert!(replica.stopped().await, "the task ends at the stop");
    assert!(!replica.overloaded(A), "a verdict stands behind the stop");
    let polled = replica.polls();
    time::sleep(EVERY * 3).await;
    assert_eq!(replica.polls(), polled, "a poll went out after the stop");
}

/// Once the word's teller is gone nothing leads here: the monitor is
/// stopped and its verdicts are taken off, and its task still ends at
/// the stop alone.
#[tokio::test(start_paused = true)]
async fn a_word_with_no_teller_stops_the_monitor() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    // Nine wait: over the eight of the program's default.
    replica.engine.waiting.store(9, SeqCst);
    replica.role.send_replace(Some(1));
    assert!(comes(|| replica.overloaded(A)).await);

    let (gone, _) = watch::channel(None);
    drop(std::mem::replace(&mut replica.role, gone));
    time::sleep(EVERY).await;
    assert!(!replica.overloaded(A), "a verdict stands with no teller");
    let polled = replica.polls();
    time::sleep(EVERY * 3).await;
    assert_eq!(replica.polls(), polled, "a poll went out with no teller");
    assert_eq!(replica.task.len(), 1, "the task waits for the stop");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A poll in flight at the lead's end ends with its task: its answer
/// never lands, so it publishes no report and writes no verdict behind
/// the stop.
#[tokio::test(start_paused = true)]
async fn a_poll_in_flight_at_the_leads_end_lands_nothing() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    replica.role.send_replace(Some(1));
    assert!(comes(|| replica.waiting(A) == Some(0)).await);

    // The next poll answers a second late, with nine waiting: over the
    // eight of the program's default.
    replica.engine.waiting.store(9, SeqCst);
    replica.engine.late.store(1_000, SeqCst);
    time::sleep(EVERY).await;
    assert!(comes(|| replica.polls() == 2).await, "the next poll is out");
    replica.role.send_replace(None);
    assert!(
        comes(|| replica.waiting(A).is_none()).await,
        "the stop publishes the snapshot empty"
    );

    time::sleep(Duration::from_secs(2)).await;
    assert_eq!(replica.waiting(A), None, "the poll in flight landed");
    assert!(!replica.overloaded(A), "the poll in flight wrote a verdict");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// Two groups whose polls each outlast the period keep a dropped
/// monitor: each group's loop takes it again while the other's poll
/// holds it. The stop ends such a monitor at the lead's end: no poll
/// goes out behind it, and the next lead starts it at once.
#[tokio::test(start_paused = true)]
async fn a_monitor_its_own_polls_would_keep_is_stopped_at_the_leads_end() {
    let mut replica = Replica::new();
    replica.register(facts(A));
    replica.register(serving(B, "n"));
    // Each answer comes behind the period, inside the poll's deadline.
    replica.engine.late.store(2_500, SeqCst);
    replica.role.send_replace(Some(1));
    assert!(
        comes(|| replica.polls() == 2).await,
        "each group's first poll is out"
    );

    replica.role.send_replace(None);
    time::sleep(Duration::from_millis(1)).await;
    let polled = replica.polls();
    time::sleep(EVERY * 4).await;
    assert_eq!(replica.polls(), polled, "a poll went out after the lead");

    replica.role.send_replace(Some(2));
    assert!(
        comes(|| replica.polls() == polled + 2).await,
        "the next lead polls each group at once"
    );
    assert!(replica.stopped().await, "the task ends at the stop");
}
