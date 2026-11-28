use std::time::{Duration, Instant};

use smg_worker::{
    connection::engine::{EngineConnector, HealthAnswer},
    health::port::{HealthSink, ProbeAnswer},
    identity::basic::Worker,
    registry::WorkerId as ObjectId,
    test_support::{ScriptedAnswers, ScriptedConnector},
    worker::{RuntimeType, WorkerStatus as ObjectStatus},
};
use tokio::{
    task::{yield_now, JoinSet},
    time,
};

use super::{super::lead::tests::lasting, *};
use crate::{
    engine::view::{WireModes, WorkerId},
    service::{
        controller::Config,
        elector::{ElectError, Elector},
    },
    store::{
        GatewayId, HealthPolicy, MemoryStore, Reachability, Registrant, WorkerFacts, WorkerStatus,
    },
};

const A: &str = "grpc://a:9000";
const B: &str = "grpc://b:9000";
const C: &str = "grpc://c:9000";

/// The check interval of every row here, the shortest an object counts.
const INTERVAL: Duration = Duration::from_secs(1);

/// A row of a gRPC worker under `key`: probed once an interval, readied
/// by one passed probe and taken out by `failures` failed ones.
fn facts(key: &str, failures: u32) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation: 1,
        models: vec!["m".to_owned()],
        dp_size: 1,
        wire_modes: WireModes::GRPC,
        health: HealthPolicy {
            failure_threshold: failures,
            success_threshold: 1,
            check_interval_ms: 1_000,
            timeout_ms: 1_000,
            probes: true,
        },
        ..WorkerFacts::default()
    }
}

/// What the engine answers a health call.
fn says(healthy: bool) -> HealthAnswer {
    HealthAnswer {
        healthy,
        message: String::new(),
    }
}

/// A replica: its store, its worker objects over one engine the case
/// scripts, its election of one, the role's word and its prober's task.
/// No socket and no clock of the host: the engine answers in process.
struct Replica {
    store: Arc<MemoryStore>,
    workers: Arc<WorkerObjects>,
    engine: Arc<ScriptedAnswers>,
    connector: Arc<ScriptedConnector>,
    elector: Elector,
    prober: Arc<Prober>,
    role: watch::Sender<Option<u64>>,
    stop: watch::Sender<bool>,
    task: JoinSet<()>,
}

impl Replica {
    /// A replica that does not lead yet, its engine healthy and its
    /// prober's task started.
    fn new() -> Result<Self, ElectError> {
        let store = Arc::new(MemoryStore::new());
        let engine = Arc::new(ScriptedAnswers::answering_health(Ok(says(true))));
        let connector = Arc::new(
            ScriptedConnector::default().with_engine(RuntimeType::default(), Arc::clone(&engine)),
        );
        let dials: Arc<dyn EngineConnector> = connector.clone();
        let workers = Arc::new(WorkerObjects::with_connector(
            Arc::clone(&store),
            dials,
            Config::DEFAULT_WORKER_OVERLOAD,
        ));
        let elector = Elector::alone("replica", lasting())?;
        let prober = Arc::new(Prober::new(Arc::clone(&workers), elector.fence()));
        let (role, led) = watch::channel(None);
        let (stop, stopped) = watch::channel(false);
        let mut task = JoinSet::new();
        task.spawn({
            let prober = Arc::clone(&prober);
            async move { prober.run(stopped, led).await }
        });
        Ok(Self {
            store,
            workers,
            engine,
            connector,
            elector,
            prober,
            role,
            stop,
            task,
        })
    }

    /// Writes the row of `facts` through the holder, as a registration
    /// call does.
    fn register(&self, facts: WorkerFacts) -> WorkerId {
        let holder = self.workers.holder();
        let written = holder.upsert(&self.store, facts, Registrant::Control);
        written.expect("the store takes the write").id()
    }

    /// Takes the lease: the term the fence opens under.
    fn take(&mut self) -> Result<u64, ElectError> {
        self.elector.tick(Instant::now())?;
        Ok(self.elector.fence().term())
    }

    /// Takes the lease and tells its term, as the serve's teller does
    /// at the step behind a take.
    fn lead(&mut self) -> Result<u64, ElectError> {
        let term = self.take()?;
        self.role.send_replace(Some(term));
        Ok(term)
    }

    fn status(&self, row: WorkerId) -> Option<WorkerStatus> {
        self.store.get(row).map(|row| row.status)
    }

    /// The object of `row` with its id in the registry.
    fn object(&self, row: WorkerId) -> Option<(ObjectId, Arc<dyn Worker>)> {
        let holder = self.workers.holder();
        let id = holder.object(row)?;
        let worker = holder.registry().get(&id)?;
        Some((id, worker))
    }

    fn object_status(&self, row: WorkerId) -> Option<ObjectStatus> {
        self.object(row).map(|(_, worker)| worker.status())
    }

    /// One outcome through `adapter`, as a loop says it of the object of
    /// `row` as it stands.
    fn judged(
        &self,
        adapter: &HealthAdapter<Members>,
        row: WorkerId,
        passed: bool,
    ) -> Option<ProbeAnswer> {
        let (id, worker) = self.object(row)?;
        Some(adapter.probe_outcome(&id, passed, worker.revision(), worker.status()))
    }

    /// Tells the stop: whether the prober's task ended clean.
    async fn stopped(mut self) -> bool {
        self.stop.send_replace(true);
        matches!(self.task.join_next().await, Some(Ok(())))
    }
}

/// Lets the replica's tasks run, the clock standing still, until
/// `holds`: whether it came to hold.
async fn comes(mut holds: impl FnMut() -> bool) -> bool {
    for _ in 0..64 {
        if holds() {
            return true;
        }
        yield_now().await;
    }
    holds()
}

#[tokio::test(start_paused = true)]
async fn a_leaders_probe_readies_a_pending_row() {
    let mut replica = Replica::new().expect("a replica");
    replica.lead().expect("a term");
    let row = replica.register(facts(A, 1));
    assert_eq!(replica.status(row), Some(WorkerStatus::Pending));

    assert!(
        comes(|| replica.status(row) == Some(WorkerStatus::Ready)).await,
        "the row's first passed probe readies it"
    );
    assert_eq!(replica.object_status(row), Some(ObjectStatus::Ready));
    assert_eq!(replica.connector.connects(), 1);
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A ready row leaves by its policy's count of failed probes, one an
/// interval, and not before.
#[tokio::test(start_paused = true)]
async fn a_failed_probe_moves_the_row_by_its_policys_threshold() {
    let mut replica = Replica::new().expect("a replica");
    replica.lead().expect("a term");
    let row = replica.register(facts(A, 3));
    assert!(comes(|| replica.status(row) == Some(WorkerStatus::Ready)).await);

    replica.engine.set_health(Ok(says(false)));
    time::sleep(INTERVAL * 2 + INTERVAL / 2).await;
    assert_eq!(
        replica.status(row),
        Some(WorkerStatus::Ready),
        "two failed probes under a threshold of three"
    );
    time::sleep(INTERVAL).await;
    assert_eq!(replica.status(row), Some(WorkerStatus::NotReady));
    assert_eq!(replica.object_status(row), Some(ObjectStatus::NotReady));
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A probe removes no row and gives up on none: a row that failed keeps
/// its probes and returns by the first one its engine passes.
#[tokio::test(start_paused = true)]
async fn a_failed_row_is_probed_in_place_and_returns_when_its_engine_answers() {
    let mut replica = Replica::new().expect("a replica");
    replica.lead().expect("a term");
    let row = replica.register(facts(A, 1));
    assert!(comes(|| replica.status(row) == Some(WorkerStatus::Ready)).await);

    // One failed probe takes the row out, and three more fail it.
    replica.engine.set_health(Ok(says(false)));
    time::sleep(INTERVAL * 4 + INTERVAL / 2).await;
    assert_eq!(replica.status(row), Some(WorkerStatus::Failed));
    assert!(
        replica.object(row).is_some(),
        "the failed row keeps its object"
    );

    replica.engine.set_health(Ok(says(true)));
    time::sleep(INTERVAL).await;
    assert_eq!(replica.status(row), Some(WorkerStatus::Ready));
    assert_eq!(replica.object_status(row), Some(ObjectStatus::Ready));
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A replica that does not lead dials no engine for its health.
#[tokio::test(start_paused = true)]
async fn a_standby_probes_nothing() {
    let replica = Replica::new().expect("a replica");
    let row = replica.register(facts(A, 1));

    time::sleep(INTERVAL * 3).await;
    assert_eq!(replica.connector.connects(), 0, "a standby dialed");
    assert_eq!(replica.status(row), Some(WorkerStatus::Pending));
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A row that arrives while nobody leads here waits for the next term,
/// whose loop probes it at once and writes under that term. A word that
/// goes from one term to the next is another lead.
#[tokio::test(start_paused = true)]
async fn a_terms_end_stops_the_probes_and_the_next_term_starts_them() {
    let mut replica = Replica::new().expect("a replica");
    let first = replica.lead().expect("the first term");
    let a = replica.register(facts(A, 1));
    assert!(comes(|| replica.status(a) == Some(WorkerStatus::Ready)).await);

    assert!(replica.elector.release().expect("a release"));
    replica.role.send_replace(None);
    time::sleep(INTERVAL).await;
    let b = replica.register(facts(B, 1));
    time::sleep(INTERVAL * 3).await;
    assert_eq!(
        replica.connector.connects(),
        1,
        "a row was dialed after the term's end"
    );
    assert_eq!(replica.status(b), Some(WorkerStatus::Pending));

    let second = replica.lead().expect("the second term");
    assert!(second > first);
    assert!(
        comes(|| replica.status(b) == Some(WorkerStatus::Ready)).await,
        "the next term's loop probes the row and writes under its term"
    );

    // A release and the take behind it, told as one: the word goes
    // from the last term to the next, and none between them says that
    // nobody leads here.
    assert!(replica.elector.release().expect("a release"));
    assert!(replica.lead().expect("the third term") > second);
    let c = replica.register(facts(C, 1));
    assert!(
        comes(|| replica.status(c) == Some(WorkerStatus::Ready)).await,
        "a word that names another term starts that term's loop"
    );
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// The term ends by the role's word, and by the fence where the word
/// still stands: an outcome behind either leaves the row's status and
/// revision as they were.
#[tokio::test(start_paused = true)]
async fn an_outcome_behind_its_terms_end_is_not_written() {
    let mut replica = Replica::new().expect("a replica");
    let row = replica.register(facts(A, 1));
    let term = replica.take().expect("a term");
    // The term's adapter with a word of the case's own: the replica's
    // word names no term, so no loop probes beside the outcomes below.
    let (word, led) = watch::channel(Some(term));
    let lead: Arc<dyn Lead> = Arc::new(ElectedLead::new(replica.elector.fence(), led));
    let adapter = replica.prober.adapter(lead, term);

    assert_eq!(
        replica.judged(&adapter, row, true),
        Some(ProbeAnswer::Moved {
            old: ObjectStatus::Pending,
            new: ObjectStatus::Ready
        }),
        "under its term the adapter writes"
    );
    let stood = replica.store.get(row).expect("the row");
    let stands = |replica: &Replica| {
        replica
            .store
            .get(row)
            .is_some_and(|row| (row.status, row.revision) == (stood.status, stood.revision))
    };

    word.send_replace(None);
    assert_eq!(
        replica.judged(&adapter, row, false),
        Some(ProbeAnswer::Stale)
    );
    assert!(stands(&replica), "an outcome behind the word's turn");

    word.send_replace(Some(term));
    assert!(replica.elector.release().expect("a release"));
    assert_eq!(
        replica.judged(&adapter, row, false),
        Some(ProbeAnswer::Stale)
    );
    assert!(stands(&replica), "an outcome behind the fence's close");
    assert!(replica.stopped().await, "the task ends at the stop");
}

#[tokio::test(start_paused = true)]
async fn a_row_whose_policy_turns_probes_off_is_not_probed() {
    let mut replica = Replica::new().expect("a replica");
    replica.lead().expect("a term");
    let mut unprobed = facts(A, 1);
    unprobed.health.probes = false;
    let off = replica.register(unprobed);
    let on = replica.register(facts(B, 1));

    time::sleep(INTERVAL * 3).await;
    assert_eq!(
        replica.status(on),
        Some(WorkerStatus::Ready),
        "the row beside it is probed"
    );
    assert_eq!(replica.connector.connects(), 1, "one row was dialed");
    assert_eq!(replica.status(off), Some(WorkerStatus::Pending));
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A row one gateway alone reaches is not the replica's to dial: its
/// object stands unchecked, so the loop sends it no probe and no
/// verdict moves its row, whatever its policy says.
#[tokio::test(start_paused = true)]
async fn a_row_one_gateway_alone_reaches_is_not_probed() {
    let mut replica = Replica::new().expect("a replica");
    replica.lead().expect("a term");
    let local = replica.register(WorkerFacts {
        reachable_from: Reachability::Gateway(GatewayId(1)),
        ..facts(A, 1)
    });
    let dialed = replica.register(facts(B, 1));

    time::sleep(INTERVAL * 3).await;
    assert_eq!(
        replica.status(dialed),
        Some(WorkerStatus::Ready),
        "the row beside it is probed"
    );
    assert_eq!(replica.connector.connects(), 1, "one row was dialed");
    assert_eq!(replica.status(local), Some(WorkerStatus::Pending));
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// A row whose facts move has its object replaced in its place. The
/// loop probes the new object at once, with no interval passed, and
/// its outcome is judged under the row's new revision.
#[tokio::test(start_paused = true)]
async fn a_replaced_object_is_probed_in_the_place_of_the_old_one() {
    let mut replica = Replica::new().expect("a replica");
    replica.lead().expect("a term");
    let row = replica.register(facts(A, 1));
    assert!(comes(|| replica.status(row) == Some(WorkerStatus::Ready)).await);
    let (place, old) = replica.object(row).expect("the row's object");

    replica.engine.set_health(Ok(says(false)));
    let mut moved = facts(A, 1);
    moved.labels.insert("pool".to_owned(), "b".to_owned());
    assert_eq!(replica.register(moved), row, "the same row, other facts");
    assert!(
        comes(|| replica.status(row) == Some(WorkerStatus::NotReady)).await,
        "the new object's probe is judged"
    );
    let (stands, new) = replica.object(row).expect("the row's object");
    assert_eq!(stands, place, "the replacement keeps its place");
    assert!(!Arc::ptr_eq(&old, &new), "another object stands there");
    assert!(replica.stopped().await, "the task ends at the stop");
}

/// With the word's teller gone nothing leads here: a loop that runs
/// ends, and the task stays until the stop, as the serve expects of
/// each of its tasks.
#[tokio::test(start_paused = true)]
async fn a_closed_word_ends_the_probes_and_the_task_stays_until_the_stop() {
    for leads in [true, false] {
        let mut replica = Replica::new().expect("a replica");
        if leads {
            replica.lead().expect("a term");
            let a = replica.register(facts(A, 1));
            assert!(comes(|| replica.status(a) == Some(WorkerStatus::Ready)).await);
        }

        let (gone, _) = watch::channel(None);
        drop(std::mem::replace(&mut replica.role, gone));
        time::sleep(INTERVAL).await;
        let b = replica.register(facts(B, 1));
        let early = time::timeout(INTERVAL * 3, replica.task.join_next()).await;
        assert!(early.is_err(), "the task ended before the stop");
        assert_eq!(
            replica.connector.connects(),
            usize::from(leads),
            "a row was dialed behind the closed word"
        );
        assert_eq!(replica.status(b), Some(WorkerStatus::Pending));
        assert!(replica.stopped().await, "the task ends at the stop");
    }
}
