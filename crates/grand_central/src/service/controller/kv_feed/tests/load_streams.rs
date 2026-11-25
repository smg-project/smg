//! The stock controller's cases of the stream a replica opens for the load
//! records alone of a gRPC worker that is not indexed. The serve hands its
//! follow the load monitor, which says whose loads are wanted
//! ([`KvFeed::set_loads_wanted`]), so every replica opens that stream.

use std::{rc::Rc, time::Instant};

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use smg_worker::{identity::basic::Worker, kv_events::KvEventCollector, load::state::LoadReceiver};

use super::*;
use crate::{
    service::{
        controller::{
            config::Config,
            election::{self, ElectionConfig, ElectionObservation, LeasePlace, Running},
            metrics::LoadTotals,
            readiness::Readiness,
        },
        elector::ElectionSettings,
    },
    worker_host::KvHost,
};

/// The worker crate's count of the streams that connected to a worker.
const STREAMS: &str = "smg_kv_event_subscriptions_total";
/// Its count of the batches a worker's streams brought, by what became of
/// each.
const BATCHES: &str = "smg_kv_event_batches_total";
/// The label of a batch that carried a load record alone.
const LOAD_ONLY: &str = "disposition=\"load_only\"";
/// The version the load monitor writes on the report of a stream's record:
/// a poll's report states its engine's own.
const PUSHED: &str = "pushed";
/// The whole of a case: the serve's start, the case and the serve's stop.
const WHOLE: Duration = Duration::from_secs(30);
/// The election's timing where a case waits for a take: a candidate looks
/// every 50 ms, and a holder keeps its lease through a pause of a second
/// on a loaded host.
const QUICK: ElectionSettings = ElectionSettings {
    duration: Duration::from_millis(1_500),
    renew_period: Duration::from_millis(50),
    renew_deadline: Duration::from_secs(1),
};

/// What a case reads of a replica of the stock controller, taken ahead of
/// its serve, and the mock worker the replica dials: a gRPC worker whose
/// record asks for no cache events, so it is not indexed.
struct Replica {
    url: String,
    record: WorkerFacts,
    store: Arc<MemoryStore>,
    objects: Arc<Objects<Members>>,
    collector: Arc<KvEventCollector<KvHost<Members>>>,
    readiness: Readiness,
    loads: LoadReceiver,
    counts: Box<dyn Fn() -> LoadTotals>,
    series: PrometheusHandle,
}

impl Replica {
    /// Writes the worker's record as `Ready` through the store's own
    /// entries and has the holder follow it, as a host that writes its
    /// store does: no task of the serve lays the record's object.
    fn lay(&self) {
        let id = self.store.upsert(self.record.clone()).id();
        let revision = self.store.get(id).expect("the record").revision;
        self.store
            .set_status(id, WorkerStatus::Ready, revision)
            .expect("the worker is ready");
        self.objects.follow(&*self.store);
        self.store.publish_now();
    }

    /// Waits for `holds`, which `what` names.
    async fn until(&self, what: &str, holds: impl Fn() -> bool) {
        let held = timeout(WAIT * 2, async {
            while !holds() {
                sleep(POLL).await;
            }
        });
        assert!(held.await.is_ok(), "{what}");
    }

    /// Waits for the replica's state to be `state`.
    async fn becomes(&self, state: State) {
        let what = format!("the replica's state is {}", state.as_str());
        self.until(&what, || self.readiness.state() == state).await;
    }

    /// The worker's object.
    fn worker(&self) -> Arc<dyn Worker> {
        let worker = self.objects.registry().get_by_url(&self.url);
        worker.expect("the record's object")
    }

    /// Whether the worker's object is marked as one whose stream pushes
    /// its load: the monitor marks it at the first record it takes, and a
    /// poll marks none.
    fn pushes(&self) -> bool {
        let state = self.worker().runtime_state();
        state.is_some_and(|state| state.pushes_load_records())
    }

    /// The version the snapshot's report of the worker states: none where
    /// the snapshot holds no report of it.
    fn version(&self) -> Option<String> {
        let snapshot = self.loads.borrow();
        snapshot.get(&self.url).map(|report| report.version.clone())
    }

    /// The worker's count in the series `name` under `label`: zero where
    /// the series does not stand.
    fn count(&self, name: &str, label: &str) -> u64 {
        let worker = format!("worker=\"{}\"", self.url);
        let open = format!("{name}{{");
        let rendered = self.series.render();
        let read = rendered.lines().find_map(|line| {
            let (series, value) = line.rsplit_once(' ')?;
            let named = series.starts_with(&open) && series.contains(&worker);
            let counted = named && series.contains(label);
            counted.then(|| value.parse().ok()).flatten()
        });
        read.unwrap_or(0)
    }
}

/// Runs `case` over a replica of the stock controller under `config` and
/// the mock worker it dials. The worker serves a stream of load records
/// alone where it is `realistic`, and refuses one as unimplemented where
/// it is not. The worker crate's series are recorded for the case's
/// thread, which runs the streams' tasks.
async fn served<F: Future<Output = ()>>(
    config: Config,
    realistic: bool,
    case: impl FnOnce(Rc<Replica>) -> F,
) {
    let recorder = PrometheusBuilder::new().build_recorder();
    let series = recorder.handle();
    let _recording = metrics::set_default_local_recorder(&recorder);
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("the mock worker's listener binds");
    let addr = listener.local_addr().expect("the mock worker's address");
    let url = format!("grpc://{addr}");
    let mock = MockConfig {
        model_id: MODEL.to_owned(),
        realistic,
        ..MockConfig::default()
    };
    let record = WorkerFacts {
        engine_kind: mock.grpc_wire.as_str().to_owned(),
        ..facts(&url, WireModes::GRPC, false)
    };
    let (stop_mock, mock_stopped) = oneshot::channel();
    let mut tasks = JoinSet::new();
    tasks.spawn(grpc::serve_with_listener_until(
        Arc::new(mock),
        listener,
        async {
            let _ = mock_stopped.await;
        },
    ));

    let controller = Controller::bind(config)
        .await
        .expect("the controller binds");
    let replica = Rc::new(Replica {
        url,
        record,
        store: controller.store(),
        objects: controller.workers.holder(),
        collector: Arc::clone(controller.kv_feed.collector()),
        readiness: controller.readiness(),
        loads: controller.monitor.loads(),
        counts: Box::new(controller.monitor.counts()),
        series,
    });
    let (stop, stopped) = oneshot::channel();
    let serve = Box::pin(controller.serve(async {
        let _ = stopped.await;
    }));
    let case = async {
        case(replica).await;
        stop.send(()).expect("the controller is still serving");
    };
    let (result, ()) = timeout(WHOLE, async { tokio::join!(serve, case) })
        .await
        .expect("the case ends and the controller stops");
    result.expect("the controller stops cleanly");
    stop_mock
        .send(())
        .expect("the mock worker is still serving");
    timeout(WAIT, tasks.join_next())
        .await
        .expect("the mock worker stops")
        .expect("the mock worker's task")
        .expect("the mock worker stops cleanly");
}

/// A configuration whose lease is a file that another holder has: an
/// election thread of the case's own, which holds the lease until its
/// stop. A replica bound under it stands by until then. The lease's
/// directory stands for as long as the first value.
async fn held_by_another() -> (tempfile::TempDir, Config, Running) {
    let directory = tempfile::tempdir().expect("a directory for the lease");
    let mut config = valid();
    config.election = ElectionConfig {
        lease: LeasePlace::File(directory.path().join("lease")),
        settings: QUICK,
        advertise: None,
    };
    let other = election::elector(&config.election, "other@127.0.0.1:1".to_owned())
        .await
        .expect("the lease's directory stands");
    let fence = other.fence();
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let held = election::start(other, Arc::new(MemoryStore::new()), standing)
        .expect("the election's thread starts");
    let leads = async {
        while fence.leader_term(Instant::now()).is_none() {
            sleep(POLL).await;
        }
    };
    timeout(WAIT, leads)
        .await
        .expect("the other holder takes the lease");
    (directory, config, held)
}

/// The replica that leads reads the load of a gRPC worker that is not
/// indexed from a stream of the worker's load records alone. One stream
/// connects, a record of it stands in the snapshot the decisions read, and
/// the worker's object is marked as one whose stream pushes. The worker is
/// laid behind the take, so its group's first poll alone can go out ahead
/// of the record: the next is a poll interval away.
#[tokio::test]
async fn a_worker_that_is_not_indexed_streams_its_load_to_the_leader_ahead_of_the_next_poll() {
    served(valid(), true, |replica| async move {
        replica.becomes(State::Leader).await;
        replica.lay();
        let pushed = || replica.version().as_deref() == Some(PUSHED);
        let stands = "a record of the stream stands in the leader's snapshot";
        replica.until(stands, pushed).await;
        assert!(replica.pushes(), "the monitor marks the worker at a record");
        assert_eq!(replica.count(STREAMS, ""), 1, "one stream connects");
        let polls = (replica.counts)().polls;
        let asked = polls.polls + polls.fallbacks;
        assert!(asked <= 1, "a poll of the next tick came first: {polls:?}");
    })
    .await;
}

/// A standby holds the stream too and drops its records. Under another
/// holder of the lease the replica opens one stream to the worker, and the
/// stream's records come: the first at its start and the next a second
/// later. No report of the worker is published and its object is not
/// marked.
#[tokio::test]
async fn a_standby_holds_the_load_stream_and_drops_its_records() {
    let (_lease, config, mut other) = held_by_another().await;
    served(config, true, |replica| async move {
        replica.becomes(State::Follower).await;
        replica.lay();
        let came = || replica.count(BATCHES, LOAD_ONLY) >= 2;
        let comes = "two records of the stream come to the standby";
        replica.until(comes, came).await;
        assert_eq!(replica.count(STREAMS, ""), 1, "one stream connects");
        assert_eq!(replica.version(), None, "a standby published a report");
        assert!(!replica.pushes(), "a standby marked the worker");
        assert_eq!(replica.readiness.state(), State::Follower);
    })
    .await;
    other.stop();
    other.ended().await;
}

/// A worker that serves no stream of load records stays polled, and its
/// refusal costs it no verdict. The mock worker answers the subscription
/// as unimplemented: the stream's task ends there, and no stream connects.
/// The leader polls the worker at each tick, the report in its snapshot is
/// a poll's, and the worker's object is neither marked as one whose stream
/// pushes nor judged by the liveness sweep, which reads a failed contact.
#[tokio::test]
async fn a_worker_that_serves_no_load_stream_stays_polled_and_unjudged() {
    let mut config = valid();
    config.load_poll_interval = Duration::from_secs(1);
    served(config, false, |replica| async move {
        replica.becomes(State::Leader).await;
        replica.lay();
        let refused = async {
            loop {
                let (slot, _) = replica.collector.slot_ids(&replica.url).await;
                if slot.is_some() && replica.collector.task_ended(&replica.url).await {
                    break;
                }
                sleep(POLL).await;
            }
        };
        timeout(WAIT, refused)
            .await
            .expect("the stream's task ends at the worker's refusal");
        let polled = || (replica.counts)().polls.polls >= 2;
        replica
            .until("the worker is polled at two ticks", polled)
            .await;
        assert_eq!(replica.count(STREAMS, ""), 0, "a stream connected");
        let version = replica.version();
        let polls = version.as_deref().is_some_and(|version| version != PUSHED);
        assert!(polls, "the report of a poll, not {version:?}");
        assert!(!replica.pushes(), "the monitor marked the worker");
        let verdict = replica.worker().stall_reason();
        assert!(verdict.is_none(), "the sweep judged the worker");
    })
    .await;
}

/// Behind a take the new leader reads the stream its standby opened. Under
/// another holder of the lease the replica opens the stream and drops its
/// first record. The holder leaves, the replica takes the lead, and a later
/// record of the same stream stands in its snapshot: one stream connected
/// in all.
#[tokio::test]
async fn a_take_reads_the_load_stream_its_standby_opened() {
    let (_lease, config, mut other) = held_by_another().await;
    served(config, true, |replica| async move {
        replica.becomes(State::Follower).await;
        replica.lay();
        let came = || replica.count(BATCHES, LOAD_ONLY) >= 1;
        let comes = "a record of the stream comes to the standby";
        replica.until(comes, came).await;
        assert!(!replica.pushes(), "a standby marked the worker");

        other.stop();
        replica.becomes(State::Leader).await;
        let pushed = || replica.version().as_deref() == Some(PUSHED);
        let stands = "a record of the stream stands in the new leader's snapshot";
        replica.until(stands, pushed).await;
        assert!(replica.pushes(), "the monitor marks the worker at a record");
        assert_eq!(replica.count(STREAMS, ""), 1, "the take connected a stream");
        other.ended().await;
    })
    .await;
}
