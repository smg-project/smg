//! The load reports of the replica's workers while it leads.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use smg_worker::{
    common_proto::EngineLoad,
    identity::basic::Worker,
    kv_events::PushedLoadSink,
    load::{monitor::WorkerMonitor, state::LoadReceiver},
    registry::WorkerRegistry,
};
use tokio::sync::watch;

use super::WorkerObjects;
use crate::{
    service::controller::metrics::LoadTotals,
    worker_host::{load_monitor, Members},
};

/// The load reports of the replica's workers while it leads: the worker
/// crate's own load monitor over the holder's registry, one for the
/// replica's life, started at each take of the lead and stopped at each
/// loss of it.
///
/// While the replica leads the monitor polls every `Ready` object the
/// replica dials and publishes the reports as one snapshot. It takes the
/// load records on the workers' event streams as polls
/// ([`PushedRecords`]). Its own sink writes the overload verdict on each
/// object by the thresholds the object states. It writes no row: a report
/// is read where the monitor publishes it. A row that one gateway alone
/// reaches is asked for no load.
///
/// The serve runs this task, and every deciding thread holds the monitor's
/// channel from the bind: a single request's decision is given the newest
/// snapshot on it, where its document reads reports and keeps none.
pub(crate) struct LoadMonitor {
    monitor: Arc<WorkerMonitor<Members>>,
    /// The holder's registry, which writes the monitor's overload verdicts
    /// on the objects and counts each one that changes.
    registry: Arc<WorkerRegistry<Members>>,
    /// Where the load records on the workers' event streams go: the serve
    /// hands it to its feed of those streams.
    pushed: Arc<PushedRecords>,
}

/// Where the load records on the workers' event streams go: the replica's
/// load monitor while this replica leads, and nowhere while it leads none.
///
/// The streams stand on every replica, for the cache events they carry,
/// and the worker crate's monitor takes a pushed record whether it runs or
/// not. So the replica's word is kept here. A record that comes while the
/// monitor runs is handed on, and the monitor takes it as a poll: a worker
/// whose records are fresh is polled no more. A record that comes while it
/// is stopped is dropped: a standby files no report, writes no verdict and
/// starts no task of the monitor's for it.
///
/// A stream's task hands a record on under a shared hold of the word.
/// [`LoadMonitor::run`] turns the word under the exclusive hold: it sets
/// it behind the monitor's start and clears it ahead of the monitor's
/// stop, so no record lands behind a stop. Nothing else takes either
/// hold, and no decision does.
pub(crate) struct PushedRecords {
    monitor: Arc<WorkerMonitor<Members>>,
    /// Whether the monitor runs.
    runs: RwLock<bool>,
}

impl PushedLoadSink for PushedRecords {
    fn push_load(
        self: Arc<Self>,
        worker: &Arc<dyn Worker>,
        dp_rank: i32,
        record: &EngineLoad,
        received_at: Instant,
    ) {
        let runs = self.runs.read();
        if *runs {
            self.monitor
                .apply_pushed_load(worker, dp_rank, record, received_at);
        }
    }
}

impl LoadMonitor {
    /// The monitor of `workers`, which polls each worker once an `every`
    /// while this replica leads. It is built stopped: [`Self::run`] starts
    /// it.
    pub(crate) fn new(workers: &WorkerObjects, every: Duration) -> Self {
        let holder = workers.holder();
        let monitor = load_monitor(&holder, every);
        Self {
            pushed: Arc::new(PushedRecords {
                monitor: Arc::clone(&monitor),
                runs: RwLock::new(false),
            }),
            monitor,
            registry: Arc::clone(holder.registry()),
        }
    }

    /// The sink of the load records that ride the workers' event streams,
    /// for the serve's feed of those streams: one for the replica's life,
    /// as the monitor is.
    pub(crate) fn pushed(&self) -> &Arc<PushedRecords> {
        &self.pushed
    }

    /// The monitor's channel, for the reader of its snapshots: one channel
    /// for the replica's life. The monitor publishes on it under every
    /// lead, and publishes it empty at each lead's end.
    pub(crate) fn loads(&self) -> LoadReceiver {
        self.monitor.subscribe()
    }

    /// The reader of what the monitor's polls did and of how often an
    /// overload verdict of a worker changed, for the metrics render: the
    /// worker crate's own counts, read at each call. The counts are the
    /// replica's: they stand through every stop and start of the monitor,
    /// and a lead's end, which takes every verdict off, counts each one it
    /// takes.
    pub(crate) fn counts(&self) -> impl Fn() -> LoadTotals + Send + Sync + 'static {
        let (monitor, registry) = (Arc::clone(&self.monitor), Arc::clone(&self.registry));
        move || LoadTotals {
            polls: monitor.poll_totals(),
            overload_flips: registry.overload_edges(),
        }
    }

    /// Runs the load monitor while this replica leads and stops it while it
    /// leads none, until `stop` reads true. `led` is the role's word: the
    /// term this replica leads, if it leads one. The monitor runs for the
    /// whole lead, whatever the documents say: a decision over the objects
    /// reads the monitor's verdicts off them under every policy.
    ///
    /// At the word's first term the task starts the monitor
    /// ([`WorkerMonitor::start_event_loop`]) and then lets the streams'
    /// load records in ([`PushedRecords`]). At the word that nobody leads
    /// here it shuts them out, then stops the monitor and waits for the
    /// stop's end ([`WorkerMonitor::stop`]). Every task of the monitor
    /// that stands has ended then, a poll in flight with its task. The
    /// snapshot is published empty and every overload verdict is off the
    /// objects. The monitor keeps nothing that a stream's records filed,
    /// no rank of a report and no publish that waited for its window: the
    /// next lead starts each worker from no report. So a replica that does
    /// not lead dials no engine for its load, and the next lead's start
    /// comes behind the last poll. This task alone starts and stops the
    /// monitor, each in its turn.
    ///
    /// One loop is not awaited: that of a group which a registration, a
    /// removal or a change of its interval ended just ahead of the stop. A
    /// tick of it that is past its fetch can still publish a report and
    /// write a verdict behind the stop. They stand until the next start,
    /// which clears the snapshot and takes every verdict off ahead of its
    /// first poll: a lead starts from no report, and a decision reads one of
    /// the lead before only in the instant between a take and that start.
    ///
    /// A word that goes from one term to the next leaves the monitor
    /// running. It writes nothing under a term, so a term's end needs no
    /// end of its own: an engine's report stands whoever leads.
    ///
    /// The task ends at the stop alone, behind the same stop of its
    /// monitor. Once the word's teller is gone nothing leads here: the
    /// monitor is stopped the same way and the task waits for the stop. A
    /// task that is dropped ahead of its stop stops nothing: its monitor
    /// runs on until it is dropped.
    pub(crate) async fn run(
        &self,
        mut stop: watch::Receiver<bool>,
        mut led: watch::Receiver<Option<u64>>,
    ) {
        let stopped = async {
            let _ = stop.wait_for(|stop| *stop).await;
        };
        tokio::pin!(stopped);
        loop {
            let leads = tokio::select! {
                biased;
                () = &mut stopped => return,
                told = led.wait_for(Option::is_some) => told.is_ok(),
            };
            // No word: its teller is gone.
            if !leads {
                break;
            }
            self.monitor.start_event_loop();
            *self.pushed.runs.write() = true;
            // `None` at the stop, else whether the word still has its
            // teller.
            let told = tokio::select! {
                biased;
                () = &mut stopped => None,
                told = led.wait_for(Option::is_none) => Some(told.is_ok()),
            };
            // Ahead of the stop: a record that came behind it would file a
            // report and start a task the stop no longer ends.
            *self.pushed.runs.write() = false;
            self.monitor.stop().await;
            match told {
                None => return,
                Some(true) => {}
                Some(false) => break,
            }
        }
        stopped.await;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering::SeqCst},
    };

    use async_trait::async_trait;
    use smg_worker::{
        common_proto::KvEventBatch,
        connection::engine::{
            AdminAnswer, EngineConnection, EngineConnector, HealthAnswer, ZmqLink,
        },
        worker::{ProfileOptions, RuntimeType, SchedulerLoadSnapshot, WorkerLoadResponse},
    };
    use tokio::{task::JoinSet, time};

    use super::{super::ObjectsWake, *};
    use crate::{
        engine::view::WireModes,
        service::controller::frame_loads::FrameLoads,
        store::{GatewayId, MemoryStore, Reachability, WorkerFacts, WorkerStatus, WorkersStore},
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

        async fn start_profile(
            &self,
            _options: &ProfileOptions,
        ) -> Result<AdminAnswer, tonic::Status> {
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
            let store = Arc::new(MemoryStore::new());
            let engine = Arc::new(Engine::default());
            let dials: Arc<dyn EngineConnector> = Arc::new(Dials(Arc::clone(&engine)));
            let workers = Arc::new(WorkerObjects::with_connector(
                Arc::clone(&store),
                ObjectsWake::default(),
                dials,
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

        /// Writes the `Ready` row of `facts` and lays its object, as the
        /// follower's task does at the store's wake.
        fn register(&self, facts: WorkerFacts) {
            let row = self.store.upsert(facts).id();
            let standing = self.store.get(row).expect("the row");
            self.store
                .set_status(row, WorkerStatus::Ready, standing.revision);
            self.workers.holder().follow(&*self.store);
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

    /// A replica that does not lead dials no engine for its load.
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
    /// it would start a publishing task that the stop no longer ends, and
    /// a standby would publish its report.
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
        *replica.engine.at_cut.lock() = Some(AtCut(Box::new(move || {
            let record = EngineLoad {
                waiting_requests: 9,
                ..EngineLoad::default()
            };
            Arc::clone(&pushed).push_load(&object, 0, &record, Instant::now());
        })));
        replica.role.send_replace(None);
        time::sleep(EVERY).await;
        assert_eq!(replica.waiting(A), None, "a record landed inside the stop");
        assert!(!replica.overloaded(A), "a verdict stands behind the stop");
        // The step holds the sink, and the engine holds the step.
        replica.engine.at_cut.lock().take();
        assert!(replica.stopped().await, "the task ends at the stop");
    }

    /// Nothing that a stream filed under one lead is published under a later
    /// one. Under the first lead B's rank 1 pushed and was published, and
    /// A's record came inside the lead's last publish window: it is still
    /// queued where the lead ends. The monitor's stop evicts both files.
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

        // Five wait on B's rank 1, and its window publishes them.
        replica.push_rank(B, 1, 5);
        assert!(
            comes(|| replica.waiting(B) == Some(5)).await,
            "the window publishes B's rank 1"
        );
        // A's record, and the lead ends inside its publish window.
        replica.push(A, 9);
        replica.role.send_replace(None);
        assert!(
            comes(|| replica.waiting(A).is_none() && replica.waiting(B).is_none()).await,
            "the lead's end empties the snapshot"
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
    /// that stood before its term at once, and one that comes during it
    /// with its group's next poll, an interval later.
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

        replica.register(facts(B));
        let polled = replica.polls();
        replica.engine.waiting.store(5, SeqCst);
        time::sleep(EVERY - Duration::from_millis(100)).await;
        assert_eq!(replica.polls(), polled, "a poll came inside the period");
        assert!(
            comes(|| replica.waiting(A) == Some(5) && replica.waiting(B) == Some(5)).await,
            "the period's poll takes both workers"
        );
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
}
