//! What no case outside the module reaches: a serve whose election
//! thread is gone on its own, one whose deciding thread unwound, and a
//! serve that leaves by a listener's error.

use std::{
    fs::File,
    future::pending,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use axum::serve::Listener as _;
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot, watch},
    task::JoinSet,
    time::timeout,
};
use tonic::{
    codegen::tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt as _},
    transport::{server::TcpIncoming, Server},
};

use super::{
    config::tests::valid,
    decide,
    decision_slots::{tests as bed, TERM_MARK},
    election::{self, ElectionObservation},
    following,
    principal::{tests::told, Layer},
    readiness::{
        tests::{detached, fenced},
        Phase,
    },
    standby::Lag,
    tell_readiness, tell_role, unnamed_port, Bound, ConfigError, Controller, Error, NoFollow,
    Scope, Service,
};
use crate::{
    service::{
        elector::{
            ConditionalWriteOutcome, ElectError, ElectionSettings, Elector, LeaseBackend,
            LeaseRecord, MemoryLease, TickOutcome,
        },
        proto::{self, busy::Reason, grand_central_server::GrandCentral as _},
    },
    store::{HealthPolicy, Registrant, WorkerFacts, WorkerStatus, WorkersStore},
};

/// A lease that stores `writes` writes: the thread that steps it panics
/// at the next.
struct GivesWay {
    lease: MemoryLease,
    writes: u32,
}

impl LeaseBackend for GivesWay {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        assert!(self.writes > 0, "the lease gave way");
        self.writes -= 1;
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        self.lease.place()
    }
}

/// Every socket the controller accepts sends at once, read from the
/// socket itself: one accepted by each gRPC listener as bound, and one
/// by the admin listener. A socket accepted with no option set does not.
#[tokio::test]
async fn every_accepted_socket_sends_at_once() {
    let plain = TcpListener::bind("127.0.0.1:0").await.expect("a port");
    let addr = plain.local_addr().expect("its address");
    let dialed = TcpStream::connect(addr).await.expect("a connection");
    let (socket, _) = plain.accept().await.expect("accepted");
    assert!(
        !socket.nodelay().expect("the option reads"),
        "the kernel's default holds small writes"
    );
    drop(dialed);

    let mut controller = Controller::bind(valid()).await.expect("the listeners bind");
    assert!(!controller.listeners.is_empty());
    for bound in &mut controller.listeners {
        let dialed = TcpStream::connect(bound.addr).await.expect("a connection");
        let socket = bound
            .listener
            .next()
            .await
            .expect("a socket")
            .expect("accepted");
        assert!(
            socket.nodelay().expect("the option reads"),
            "{}",
            bound.name
        );
        drop(dialed);
    }
    let dialed = TcpStream::connect(controller.admin_addr)
        .await
        .expect("a connection");
    let (socket, _) = controller.admin.accept().await;
    assert!(socket.nodelay().expect("the option reads"), "admin");
    drop(dialed);
}

/// The deciding threads are the configuration's count: a model's
/// requests go to one of that many.
#[tokio::test]
async fn the_controller_starts_the_deciding_threads_its_configuration_counts() {
    let mut config = valid();
    config.decide_shards = 3;
    let controller = Controller::bind(config).await.expect("the listeners bind");
    let deciders = &controller.shared.deciders;
    let highest = (0..64)
        .map(|index| deciders.of(&format!("model-{index}")))
        .max();
    assert_eq!(highest, Some(2));
}

/// The bind draws its log's first place: two binds of one configuration
/// start apart and under no term, so a place of the run before is no
/// place of this run.
#[tokio::test]
async fn the_bind_draws_its_logs_first_place() {
    let mut starts = Vec::new();
    for _ in 0..2 {
        let controller = Controller::bind(valid()).await.expect("the listeners bind");
        starts.push(controller.shared.feed.log.head());
    }
    assert_ne!(starts[0].seq, starts[1].seq, "two runs start apart");
    assert!(starts.iter().all(|start| start.seq != 0 && start.term == 0));
}

/// The thread gone at its take, before the serve read ready, and at its
/// first renewal, after: the serve ends with the error either way, so
/// the process exits and its supervisor starts it again.
#[tokio::test]
async fn a_serve_whose_election_thread_is_gone_ends_with_an_error() {
    let settings = ElectionSettings {
        duration: Duration::from_millis(300),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(200),
    };
    for writes in [0, 1] {
        let mut controller = Controller::bind(valid()).await.expect("the listeners bind");
        controller.elector = Elector::new(
            Box::new(GivesWay {
                lease: MemoryLease::new(),
                writes,
            }),
            "controller-0@127.0.0.1:1",
            settings,
        )
        .expect("the settings are in order");
        let served = timeout(Duration::from_secs(10), controller.serve(pending()))
            .await
            .expect("the serve ends within ten seconds");
        assert!(
            matches!(served, Err(Error::ElectionEnded)),
            "a lease that gave way after {writes} writes: {served:?}"
        );
    }
}

/// The serve hands its deciders the planned hand-over: a controller
/// alone over its lease leaves at its term's mark, takes its next term
/// at the election's next step and decides under it.
#[tokio::test]
async fn a_served_controller_decides_under_its_next_term_past_a_terms_mark() {
    let controller = Controller::bind(valid()).await.expect("the listeners bind");
    let (shared, store) = (Arc::clone(&controller.shared), controller.store());
    let workers = controller.workers();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let case = async {
        let (wait, begun) = (Duration::from_secs(20), Instant::now());
        while shared.fence.leader_term(Instant::now()).is_none() {
            assert!(begun.elapsed() < wait, "no first take");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        bed::fleet(&store, &workers);
        let applied = store.apply(bed::preset("random", 1));
        applied.expect("above the unbound start");
        store.spend_ids(u32::MAX - TERM_MARK + 4);
        let (sender, mut answers) = mpsc::channel(64);
        let mut terms = Vec::new();
        while !terms.contains(&2) {
            assert!(begun.elapsed() < wait, "no pick under the next term");
            let read = bed::burst(&shared.deciders, &sender, &mut answers).await;
            terms.extend(read.into_iter().flatten().map(|id| id >> 32));
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        terms.dedup();
        assert_eq!(terms, [1, 2], "the term of the mark, then the next");
        stop.send(()).expect("the serve waits for its stop");
    };
    let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, case) })
        .await
        .expect("the serve ends at its stop");
    served.expect("a serve that leaves by its signal");
}

/// The serve's presence sweep runs the upkeep of the engine its store
/// decides by, which runs no task of its own, and the store tells that
/// engine each worker that leaves. Served picks write the record of the
/// prefixes sent for their model. Once the model's last worker left,
/// the record is gone within the sweep's period, with no request behind
/// the leaves.
#[tokio::test]
async fn a_served_controller_keeps_no_sent_prefixes_of_a_model_whose_workers_left() {
    let controller = Controller::bind(valid()).await.expect("the listeners bind");
    let (shared, store) = (Arc::clone(&controller.shared), controller.store());
    let workers = controller.workers();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let case = async {
        let (wait, begun) = (Duration::from_secs(20), Instant::now());
        while shared.fence.leader_term(Instant::now()).is_none() {
            assert!(begun.elapsed() < wait, "no first take");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let (singles, sides) = bed::fleet(&store, &workers);
        let applied = store.apply(bed::preset("cache_aware", 1));
        applied.expect("above the unbound start");
        let records = || store.deciding(|world| world.front().affinity().held());
        let (sender, mut answers) = mpsc::channel(64);
        let read = bed::burst_of(&shared.deciders, &sender, &mut answers, bed::hashed).await;
        assert!(read.iter().any(Result::is_ok), "no pick: {read:?}");
        let sent = records();
        assert!(
            matches!(sent.as_slice(), [(_, workers)] if !workers.is_empty()),
            "one model's record, of the workers its picks named: {sent:?}"
        );
        for id in singles.into_iter().chain(sides.map(|(_, id)| id)) {
            let revision = store.get(id).expect("the row").revision;
            workers.remove(&store, id, revision).expect("a removal");
        }
        while !records().is_empty() {
            assert!(begun.elapsed() < wait, "a record no worker is left in");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        stop.send(()).expect("the serve waits for its stop");
    };
    let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, case) })
        .await
        .expect("the serve ends at its stop");
    served.expect("a serve that leaves by its signal");
}

/// A served controller hands its holder ([`Controller::workers`]): a
/// worker written through it while the controller leads has its object
/// when the write returns, and a worker removed has none. A row written
/// through the store's own entry gets no object: no task follows the
/// store's table.
#[tokio::test]
async fn a_served_controller_lays_a_worker_written_through_its_holder() {
    let controller = Controller::bind(valid()).await.expect("the listeners bind");
    let (shared, store) = (Arc::clone(&controller.shared), controller.store());
    let workers = controller.workers();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let case = async {
        let (wait, begun) = (Duration::from_secs(20), Instant::now());
        while shared.fence.leader_term(Instant::now()).is_none() {
            assert!(begun.elapsed() < wait, "no first take");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let facts = |url: &str| WorkerFacts {
            key: url.to_owned(),
            engine: url.to_owned(),
            models: vec!["m".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        };
        let written = workers.upsert(&store, facts("http://w:8000"), Registrant::Control);
        let row = written.expect("the store took the registration").id();
        assert!(
            workers.object(row).is_some(),
            "the object stands at the write"
        );
        let revision = store.get(row).expect("the row").revision;
        workers.remove(&store, row, revision).expect("a removal");
        assert!(workers.is_empty(), "the object left with its row");
        // The store's own entry: a row, and no object.
        let around = store.upsert(facts("http://around:8000")).id();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(workers.object(around), None, "no task follows the table");
        stop.send(()).expect("the serve waits for its stop");
    };
    let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, case) })
        .await
        .expect("the serve ends at its stop");
    served.expect("a serve that leaves by its signal");
}

/// The serve starts the probes of the replica's workers: a worker
/// written while the controller leads is readied by the first probe its
/// engine passes, with no hand on its status.
#[tokio::test]
async fn a_served_controller_probes_its_rows_while_it_leads() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
    let addr = listener.local_addr().expect("the port's address");
    let engine = axum::Router::new().route("/health", axum::routing::get(|| async {}));
    let mut engines = JoinSet::new();
    engines.spawn(async move { axum::serve(listener, engine).await });

    let controller = Controller::bind(valid()).await.expect("the listeners bind");
    let (shared, store) = (Arc::clone(&controller.shared), controller.store());
    let workers = controller.workers();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let case = async {
        let (wait, begun) = (Duration::from_secs(20), Instant::now());
        while shared.fence.leader_term(Instant::now()).is_none() {
            assert!(begun.elapsed() < wait, "no first take");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let url = format!("http://{addr}");
        let stated = WorkerFacts {
            key: url.clone(),
            engine: url,
            models: vec!["m".to_owned()],
            dp_size: 1,
            health: HealthPolicy {
                success_threshold: 1,
                ..HealthPolicy::default()
            },
            ..WorkerFacts::default()
        };
        let written = workers.upsert(&store, stated, Registrant::Control);
        let row = written.expect("the store took the registration").id();
        assert_eq!(
            store.get(row).map(|row| row.status),
            Some(WorkerStatus::Pending)
        );
        while store.get(row).map(|row| row.status) != Some(WorkerStatus::Ready) {
            assert!(begun.elapsed() < wait, "no probe readied the row");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        stop.send(()).expect("the serve waits for its stop");
    };
    let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, case) })
        .await
        .expect("the serve ends at its stop");
    served.expect("a serve that leaves by its signal");
}

/// The serve starts the restores' clock: a restore whose stream took no
/// part for a minute is let go at the clock's next beat, with no look
/// of that stream and no restore behind it.
#[tokio::test]
async fn a_served_controller_lets_a_stalled_restore_go_by_its_clock() {
    let controller = Controller::bind(valid()).await.expect("the listeners bind");
    let shared = Arc::clone(&controller.shared);
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let case = async {
        let (wait, begun) = (Duration::from_secs(20), Instant::now());
        // A restore held since a minute ago: no stream looks at it and
        // no restore comes behind it.
        let since = begun
            .checked_sub(Duration::from_secs(60))
            .expect("the clock is past its first minute");
        shared.replication.stalled_since(since);
        assert_eq!(shared.replication.held(), 1);
        while shared.replication.held() != 0 {
            assert!(begun.elapsed() < wait, "the clock let nothing go");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        stop.send(()).expect("the serve waits for its stop");
    };
    let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, case) })
        .await
        .expect("the serve ends at its stop");
    served.expect("a serve that leaves by its signal");
}

/// A deciding thread that unwound leaves its models with no decider:
/// the serve ends with the error, told before the serve began, before
/// it read ready or after, so the process exits and its supervisor
/// starts it again. A word told before the serve began is one its
/// receiver was made after: the serve reads it by its value.
#[tokio::test]
async fn a_serve_whose_deciding_thread_unwound_ends_with_an_error() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Told {
        BeforeTheServe,
        BeforeReady,
        AfterReady,
    }
    for told in [Told::BeforeTheServe, Told::BeforeReady, Told::AfterReady] {
        let controller = Controller::bind(valid()).await.expect("the listeners bind");
        let shared = Arc::clone(&controller.shared);
        let mut phase = controller.phase.subscribe();
        if told == Told::BeforeTheServe {
            shared.deciders.tell_unwound();
        }
        let unwinds = async move {
            if told == Told::AfterReady {
                let serving = |phase: &Phase| *phase == Phase::Serving;
                assert!(phase.wait_for(serving).await.is_ok(), "the serve is ready");
            }
            if told != Told::BeforeTheServe {
                shared.deciders.tell_unwound();
            }
        };
        let (served, ()) = timeout(Duration::from_secs(20), async {
            tokio::join!(controller.serve(pending()), unwinds)
        })
        .await
        .expect("the serve ends within twenty seconds");
        assert!(
            matches!(served, Err(Error::DecisionThreadEnded)),
            "told {told:?}: {served:?}"
        );
    }
}

/// A lease that takes `slow` to store a release.
struct ReleasesSlowly {
    lease: MemoryLease,
    slow: Duration,
}

impl LeaseBackend for ReleasesSlowly {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        if record.holder.is_empty() {
            thread::sleep(self.slow);
        }
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        self.lease.place()
    }
}

/// An error's leave waits for the release as a signal's does: a serve
/// that leaves by a listener's error while it leads returns that error
/// once its release is stored, not before. By order alone: the store is
/// slow to store a release, and the lease is free when the serve has
/// returned.
#[tokio::test]
async fn a_serve_that_leaves_by_an_error_while_it_leads_has_released_its_lease() {
    let mut lease = MemoryLease::new();
    let mut config = valid();
    // Room for the slow release and a loaded host inside the grace
    // period: the serve waits for the release no longer than that.
    config.shutdown_grace = Duration::from_secs(30);
    let mut controller = Controller::bind(config).await.expect("the listeners bind");
    controller.elector = Elector::new(
        Box::new(ReleasesSlowly {
            lease: lease.clone(),
            slow: Duration::from_millis(300),
        }),
        "controller-0@127.0.0.1:1",
        ElectionSettings {
            duration: Duration::from_millis(300),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(200),
        },
    )
    .expect("the settings are in order");
    let fence = controller.elector.fence();
    let (fail, failed) = oneshot::channel::<()>();
    let serve = controller.serve_until(pending(), async move {
        let _ = failed.await;
        Error::Task("a listener failed".to_owned())
    });
    let fault = async {
        // The take lands, then the listener fails.
        while fence.term() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        fail.send(()).expect("the serve waits for its fault");
    };
    let (served, ()) = timeout(Duration::from_secs(60), async {
        tokio::join!(serve, fault)
    })
    .await
    .expect("the serve ends within its grace period");
    assert!(matches!(served, Err(Error::Task(_))), "{served:?}");
    let record = lease.read().expect("a read").expect("a record");
    assert!(
        record.holder.is_empty(),
        "the lease is released when the serve has returned: {record:?}"
    );
}

/// `stream` polled by a task of its own, as a connection's task polls
/// it: at the word of the stream's waker alone. The task, which ends
/// with the case, and the words it hands on.
fn polled<S>(mut stream: S) -> (JoinSet<()>, mpsc::Receiver<S::Item>)
where
    S: Stream + Send + Unpin + 'static,
    S::Item: Send,
{
    let (words, heard) = mpsc::channel(2);
    let mut poller = JoinSet::new();
    poller.spawn(async move {
        while let Some(word) = stream.next().await {
            if words.send(word).await.is_err() {
                return;
            }
        }
    });
    (poller, heard)
}

/// The next word of a stream [`polled`] by a task of its own, waited
/// for five seconds at most.
async fn word<T>(heard: &mut mpsc::Receiver<T>) -> T {
    timeout(Duration::from_secs(5), heard.recv())
        .await
        .expect("a word within five seconds")
        .expect("the stream is open")
}

/// A watch of the rows over `gateways`, opened once its replica leads
/// and [`polled`] by a task of its own, with its first page read:
/// nothing is written behind it, and the watch waits.
async fn waiting_watch(
    gateways: &Service,
) -> (
    JoinSet<()>,
    mpsc::Receiver<Result<proto::WorkerEvent, tonic::Status>>,
) {
    let (wait, begun) = (Duration::from_secs(20), Instant::now());
    while gateways.leading(Instant::now()).is_err() {
        assert!(begun.elapsed() < wait, "no first take");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let mut watch = tonic::Request::new(proto::WatchWorkersRequest::default());
    watch.extensions_mut().insert(told("gateway-a"));
    let watching = gateways.watch_workers(watch).await.expect("opened");
    let (watcher, mut heard) = polled(watching.into_inner());
    word(&mut heard).await.expect("the rows, none yet");
    (watcher, heard)
}

/// The role's word ends with the task that tells it. A serve that
/// leaves by a listener's error ends that task with its listeners and
/// says no stop: a watch that waits for its next event is woken at the
/// word's end and ends as at a lost lead, with no wait for its
/// connection.
#[tokio::test]
async fn a_serve_that_leaves_by_an_error_ends_a_watch_that_waits() {
    let controller = Controller::bind(valid()).await.expect("the listeners bind");
    let gateways = Service::new(vec![Scope::Gateway], Arc::clone(&controller.shared));
    let (fail, failed) = oneshot::channel::<()>();
    let serve = controller.serve_until(pending(), async move {
        let _ = failed.await;
        Error::Task("a listener failed".to_owned())
    });
    let case = async {
        let (_watcher, mut heard) = waiting_watch(&gateways).await;
        // The listener fails.
        fail.send(()).expect("the serve waits for its fault");
        let ended = word(&mut heard).await.expect_err("the word's end");
        assert_eq!(ended.code(), tonic::Code::Aborted);
    };
    let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, case) })
        .await
        .expect("the serve ends within its grace period");
    assert!(matches!(served, Err(Error::Task(_))), "{served:?}");
}

/// The role's word ends with the task that tells it, and not with the
/// serve. A serve that leaves by a listener's error ends its tasks at
/// once and then waits for the release of its lease: a watch that
/// waits ends while that release is still on its way. The case holds
/// the lock of the lease's file, so no release lands until it lets go.
#[tokio::test]
async fn a_watch_that_waits_ends_ahead_of_the_release_of_a_serve_that_left_by_an_error() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut config = valid();
    config.election.lease = election::LeasePlace::File(dir.path().join("lease"));
    // Room for the held release and a loaded host inside the grace
    // period: the serve waits for the release no longer than that.
    config.shutdown_grace = Duration::from_secs(30);
    let controller = Controller::bind(config).await.expect("the listeners bind");
    let gateways = Service::new(vec![Scope::Gateway], Arc::clone(&controller.shared));
    let (fail, failed) = oneshot::channel::<()>();
    let returned = AtomicBool::new(false);
    let serve = async {
        let fault = async move {
            let _ = failed.await;
            Error::Task("a listener failed".to_owned())
        };
        let served = controller.serve_until(pending(), fault).await;
        returned.store(true, Ordering::SeqCst);
        served
    };
    let case = async {
        let (_watcher, mut heard) = waiting_watch(&gateways).await;
        // A write of the lease waits for this lock, the release too.
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.path().join("lease.lock"))
            .expect("the lock file beside the lease");
        lock.lock().expect("the lease's lock");
        fail.send(()).expect("the serve waits for its fault");
        let ended = word(&mut heard).await.expect_err("the word's end");
        assert_eq!(ended.code(), tonic::Code::Aborted);
        assert!(
            !returned.load(Ordering::SeqCst),
            "the serve still waits for its release"
        );
        lock.unlock().expect("the lock is let go");
    };
    let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, case) })
        .await
        .expect("the serve ends within its grace period");
    assert!(matches!(served, Err(Error::Task(_))), "{served:?}");
}

/// A serve that leaves by a listener's error says no close. Its return
/// ends the close's channel, whose one sender is the serve's own: a
/// decide stream still open ends there in the close's words. The
/// stream answered a request and sleeps, and a task that outlives the
/// listeners polls it.
#[tokio::test]
async fn a_serve_that_leaves_by_an_error_ends_a_decide_stream_still_open() {
    let controller = Controller::bind(valid()).await.expect("the listeners bind");
    let fence = controller.elector.fence();
    // Opened ahead of the serve: nobody leads yet, such a stream asks
    // for no session, and its request is told that nobody leads.
    let (frames, inbound) = mpsc::channel(1);
    let open = decide::serve(
        Arc::clone(&controller.shared),
        &told("gateway-a"),
        ReceiverStream::new(inbound),
    );
    let Ok(open) = open else {
        panic!("a stream opens where nobody leads");
    };
    let (_poller, mut heard) = polled(open);
    let asked = frames.send(Ok(bed::frame(1, 1))).await;
    asked.expect("the stream reads its frames");
    let answer = word(&mut heard).await.expect("an answer");
    assert_eq!(bed::busy(&answer), Reason::NotLeader);
    let (fail, failed) = oneshot::channel::<()>();
    let serve = controller.serve_until(pending(), async move {
        let _ = failed.await;
        Error::Task("a listener failed".to_owned())
    });
    let case = async {
        // The take lands, then the listener fails.
        while fence.term() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        fail.send(()).expect("the serve waits for its fault");
        let ended = word(&mut heard).await.expect_err("the stream's end");
        assert_eq!(
            (ended.code(), ended.message()),
            (tonic::Code::Unavailable, decide::CLOSING)
        );
    };
    let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, case) })
        .await
        .expect("the serve ends within its grace period");
    assert!(matches!(served, Err(Error::Task(_))), "{served:?}");
}

/// The largest grace period a caller can set is one the clock cannot
/// add to the present. A serve under it leaves as under any other, by a
/// signal and by a listener's error: it waits for the release, and the
/// lease is free when the serve has returned.
#[tokio::test]
async fn a_serve_under_a_grace_period_the_clock_cannot_hold_leaves_with_its_lease_released() {
    for by_error in [false, true] {
        let mut lease = MemoryLease::new();
        let mut config = valid();
        config.shutdown_grace = Duration::MAX;
        let mut controller = Controller::bind(config).await.expect("the listeners bind");
        controller.elector = Elector::new(
            Box::new(ReleasesSlowly {
                lease: lease.clone(),
                slow: Duration::from_millis(500),
            }),
            "controller-0@127.0.0.1:1",
            ElectionSettings {
                duration: Duration::from_millis(300),
                renew_period: Duration::from_millis(20),
                renew_deadline: Duration::from_millis(200),
            },
        )
        .expect("the settings are in order");
        let fence = controller.elector.fence();
        let (leave, left) = oneshot::channel::<()>();
        let serve = async {
            if by_error {
                controller
                    .serve_until(pending(), async move {
                        let _ = left.await;
                        Error::Task("a listener failed".to_owned())
                    })
                    .await
            } else {
                controller
                    .serve(async move {
                        let _ = left.await;
                    })
                    .await
            }
        };
        let told = async {
            // The take lands, then the serve is told to leave.
            while fence.term() == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            leave.send(()).expect("the serve waits for its leave");
        };
        let (served, ()) = timeout(Duration::from_secs(60), async { tokio::join!(serve, told) })
            .await
            .expect("the serve ends");
        if by_error {
            assert!(matches!(served, Err(Error::Task(_))), "{served:?}");
        } else {
            assert!(served.is_ok(), "{served:?}");
        }
        let record = lease.read().expect("a read").expect("a record");
        assert!(
                record.holder.is_empty(),
                "the lease is released when the serve has returned (by an error: {by_error}): {record:?}"
            );
    }
}

/// The role's one teller over a candidate's own fence: it says that the
/// replica leads once it does, that it does not at the step after a
/// loss, and at the stop that it left. The candidate is alone over its
/// lease, so each take leads at once.
#[tokio::test]
async fn the_role_is_told_at_each_change_and_as_left_at_the_stop() {
    let mut elector = Elector::alone("controller-0@127.0.0.1:1", ElectionSettings::default())
        .expect("the default settings are in order");
    let (steps, stepped) = watch::channel(ElectionObservation::default());
    let (phase, readiness) = detached(elector.fence(), stepped, Arc::new(Lag::new(Duration::ZERO)));
    let (stop, stopped) = watch::channel(false);
    let (told, mut role) = watch::channel(None);
    let mut teller = JoinSet::new();
    teller.spawn(tell_role(readiness, stopped, Vec::new(), told));
    let mut told_as = async |leads: bool| {
        let reached = timeout(
            Duration::from_secs(5),
            role.wait_for(|told| told.is_some() == leads),
        )
        .await;
        assert!(
            matches!(reached, Ok(Ok(_))),
            "the role was not told as {leads}"
        );
    };

    // Serving, and the lease taken: the step after it tells the role.
    phase.send_replace(Phase::Serving);
    assert!(matches!(
        elector.tick(Instant::now()),
        Ok(TickOutcome::Acquired { .. })
    ));
    steps.send_replace(ElectionObservation::default());
    told_as(true).await;
    // The lease let go with no stop: told at the next step.
    assert!(elector.release().expect("a release"));
    steps.send_replace(ElectionObservation::default());
    told_as(false).await;
    // Taken again, then the stop: the replica left, and the teller ends.
    assert!(matches!(
        elector.tick(Instant::now()),
        Ok(TickOutcome::Acquired { .. })
    ));
    steps.send_replace(ElectionObservation::default());
    told_as(true).await;
    stop.send_replace(true);
    told_as(false).await;
    assert!(
        teller.join_next().await.is_some(),
        "the teller ends at the stop"
    );
}

/// The role's teller sleeps to the fence's turn. A leader whose
/// renewals stop is fenced one renew deadline after the last of them,
/// by the clock alone: no step of the election stands there, and the
/// role is told at that instant all the same.
#[tokio::test]
async fn the_role_is_told_where_the_clock_alone_closes_the_fence() {
    let settings = ElectionSettings {
        duration: Duration::from_millis(1500),
        renew_period: Duration::from_millis(100),
        renew_deadline: Duration::from_millis(500),
    };
    let mut elector =
        Elector::alone("controller-0@127.0.0.1:1", settings).expect("the settings are in order");
    let (steps, stepped) = watch::channel(ElectionObservation::default());
    let (phase, readiness) = detached(elector.fence(), stepped, Arc::new(Lag::new(Duration::ZERO)));
    let (_stop, stopped) = watch::channel(false);
    let (told, mut role) = watch::channel(None);
    let mut teller = JoinSet::new();
    teller.spawn(tell_role(readiness, stopped, Vec::new(), told));
    let mut told_as = async |leads: bool| {
        let reached = timeout(
            Duration::from_secs(5),
            role.wait_for(|told| told.is_some() == leads),
        )
        .await;
        assert!(
            matches!(reached, Ok(Ok(_))),
            "the role was not told as {leads}"
        );
    };

    // The take, and two renewals ahead of the clock, each just short of
    // the deadline the one before it stamped: the fence stays open a
    // second and a half from here, so the step's telling needs no
    // narrow window.
    phase.send_replace(Phase::Serving);
    let taken = Instant::now();
    assert!(matches!(
        elector.tick(taken),
        Ok(TickOutcome::Acquired { .. })
    ));
    let mut renewed = taken;
    for _ in 0..2 {
        renewed += settings.renew_deadline - Duration::from_millis(10);
        assert!(matches!(
            elector.tick(renewed),
            Ok(TickOutcome::Renewed { .. })
        ));
    }
    steps.send_replace(ElectionObservation::default());
    told_as(true).await;
    // No renewal and no step from here: the clock closes the fence one
    // renew deadline after the last renewal, and the teller says so.
    told_as(false).await;
    assert_eq!(elector.fence().leader_term(Instant::now()), None);
}

/// A step that finds the leader standing tells the watches nothing: the
/// word of leading is sent at its change alone, so no watch's task wakes
/// once a renew period.
#[tokio::test]
async fn a_step_under_a_standing_leader_tells_the_watches_nothing() {
    async fn told_as(role: &mut watch::Receiver<Option<u64>>, leads: bool) {
        let reached = timeout(
            Duration::from_secs(5),
            role.wait_for(|told| told.is_some() == leads),
        )
        .await;
        assert!(
            matches!(reached, Ok(Ok(_))),
            "the role was not told as {leads}"
        );
    }
    let mut elector = Elector::alone("controller-0@127.0.0.1:1", ElectionSettings::default())
        .expect("the default settings are in order");
    let (steps, stepped) = watch::channel(ElectionObservation::default());
    let (phase, readiness) = detached(elector.fence(), stepped, Arc::new(Lag::new(Duration::ZERO)));
    let (_stop, stopped) = watch::channel(false);
    let (told, mut role) = watch::channel(None);
    let mut teller = JoinSet::new();
    teller.spawn(tell_role(readiness, stopped, Vec::new(), told));

    phase.send_replace(Phase::Serving);
    assert!(matches!(
        elector.tick(Instant::now()),
        Ok(TickOutcome::Acquired { .. })
    ));
    steps.send_replace(ElectionObservation::default());
    told_as(&mut role, true).await;
    // Three steps under the standing leader, each taken by the teller:
    // turns of the runtime and no clock.
    for _ in 0..3 {
        steps.send_replace(ElectionObservation::default());
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
    }
    assert!(
        !role.has_changed().expect("the teller holds the word"),
        "a step under a standing leader told the word again"
    );
    // The teller took those steps: the one after the lead's loss is told.
    assert!(elector.release().expect("a release"));
    steps.send_replace(ElectionObservation::default());
    told_as(&mut role, false).await;
}

/// A release and the take behind it, both between two reads of the
/// teller: the replica leads at each read, under another term at the
/// second. The word names that term, so a task that runs for one term
/// is told where it ended.
#[tokio::test]
async fn a_take_under_another_term_between_two_reads_is_another_word() {
    async fn names(role: &mut watch::Receiver<Option<u64>>, term: u64) {
        let told = |told: &Option<u64>| *told == Some(term);
        let reached = timeout(Duration::from_secs(5), role.wait_for(told)).await;
        assert!(matches!(reached, Ok(Ok(_))), "term {term} was not told");
    }
    let mut elector = Elector::alone("controller-0@127.0.0.1:1", ElectionSettings::default())
        .expect("the default settings are in order");
    let (steps, stepped) = watch::channel(ElectionObservation::default());
    let (phase, readiness) = detached(elector.fence(), stepped, Arc::new(Lag::new(Duration::ZERO)));
    let (_stop, stopped) = watch::channel(false);
    let (told, mut role) = watch::channel(None);
    let mut teller = JoinSet::new();
    teller.spawn(tell_role(readiness, stopped, Vec::new(), told));

    phase.send_replace(Phase::Serving);
    assert!(matches!(
        elector.tick(Instant::now()),
        Ok(TickOutcome::Acquired { .. })
    ));
    steps.send_replace(ElectionObservation::default());
    let first = elector.fence().term();
    names(&mut role, first).await;
    // The release and the take with nothing awaited between them: the
    // teller runs on this thread, so it reads the fence at neither.
    assert!(elector.release().expect("a release"));
    assert!(matches!(
        elector.tick(Instant::now()),
        Ok(TickOutcome::Acquired { .. })
    ));
    let second = elector.fence().term();
    assert_ne!(second, first, "a take is another term");
    steps.send_replace(ElectionObservation::default());
    names(&mut role, second).await;
}

/// A serve whose lead opened with no step of the election since: the
/// take of a lease with no holder to wait for leads one renew deadline
/// after its write, between two steps. The fence admits a watch from
/// that instant, and the word such a watch finds still says that nobody
/// leads here.
struct Unstepped {
    elector: Elector,
    steps: watch::Sender<ElectionObservation>,
    phase: watch::Sender<Phase>,
    /// The role's word, as a watch admitted now holds it.
    watch: watch::Receiver<Option<u64>>,
    /// The serve's stop, never said, and the teller's task.
    _serve: (watch::Sender<bool>, JoinSet<()>),
}

impl Unstepped {
    async fn new() -> Self {
        let settings = ElectionSettings {
            duration: Duration::from_millis(1500),
            renew_period: Duration::from_millis(100),
            renew_deadline: Duration::from_millis(500),
        };
        let mut elector = Elector::new(
            Box::new(MemoryLease::new()),
            "controller-0@127.0.0.1:1",
            settings,
        )
        .expect("the settings are in order");
        let (steps, stepped) = watch::channel(ElectionObservation::default());
        let (phase, readiness) =
            detached(elector.fence(), stepped, Arc::new(Lag::new(Duration::ZERO)));
        let (stop, stopped) = watch::channel(false);
        let (told, role) = watch::channel(None);
        let mut teller = JoinSet::new();
        teller.spawn(tell_role(readiness, stopped, Vec::new(), told));

        // The take, and a chain of renewals ahead of the clock, each
        // just short of the deadline the one before it stamped: the
        // lead opens one renew deadline after the take's write whatever
        // was renewed since, and the fence stays open seconds past that
        // instant, so the read below needs no narrow window. The step's
        // telling finds a follower.
        phase.send_replace(Phase::Serving);
        let taken = Instant::now();
        assert_eq!(
            elector.tick(taken).expect("the take"),
            TickOutcome::Acquired { term: 1 }
        );
        let mut renewed = taken;
        for _ in 0..10 {
            renewed += settings.renew_deadline - Duration::from_millis(10);
            assert_eq!(
                elector.tick(renewed).expect("the renewal"),
                TickOutcome::Renewed { term: 1 }
            );
        }
        steps.send_replace(ElectionObservation::default());
        let mut first = role.clone();
        first.changed().await.expect("the teller tells the step");
        assert!(first.borrow().is_none());
        // The lead opens by itself and the fence admits a watch. The
        // teller sleeps to that instant: this thread, the runtime's
        // one, is held across it with nothing awaited, so the opening
        // is told to no one ahead of the read below. The word the watch
        // finds says that nobody leads.
        thread::sleep(settings.renew_deadline);
        assert_eq!(elector.fence().leader_term(Instant::now()), Some(1));
        let mut watch = role.clone();
        assert!(watch.borrow_and_update().is_none());
        Self {
            elector,
            steps,
            phase,
            watch,
            _serve: (stop, teller),
        }
    }

    /// The watch is told again that nobody leads here.
    async fn hears(&mut self, what: &str) {
        let heard = timeout(Duration::from_secs(5), self.watch.changed()).await;
        assert!(
            matches!(heard, Ok(Ok(()))),
            "{what} was not told to a watch that stood under the word: {heard:?}"
        );
        assert!(self.watch.borrow().is_none());
    }
}

/// A watch admitted before any step told the role as leading stands
/// under the word that this replica does not lead. The drain tells that
/// word again, and the watch hears it.
#[tokio::test]
async fn a_watch_admitted_before_the_role_was_told_as_leading_hears_the_drain() {
    let mut serve = Unstepped::new().await;
    serve.phase.send_replace(Phase::Draining);
    serve.hears("the drain").await;
}

/// So does the step that finds the lead gone, with no stop.
#[tokio::test]
async fn a_watch_admitted_before_the_role_was_told_as_leading_hears_the_loss_of_its_lead() {
    let mut serve = Unstepped::new().await;
    assert!(serve.elector.release().expect("a release"));
    serve.steps.send_replace(ElectionObservation::default());
    serve.hears("the lead's loss").await;
}

/// A listener as the follow rule reads it: its scopes, its layer and
/// whether it serves TLS.
type Door<'a> = (&'a [Scope], Layer, bool);

/// The follow rule over these listeners, each bound on a port of its
/// own: the ports, and the rule's answer.
async fn followed(doors: &[Door<'_>]) -> (Vec<u16>, Result<u16, NoFollow>) {
    let mut listeners = Vec::new();
    for &(scopes, layer, tls) in doors {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        listeners.push(Bound {
            name: "listener".to_owned(),
            scopes: scopes.to_vec(),
            addr: listener.local_addr().expect("its address"),
            listener: TcpIncoming::from(listener),
            server: Server::builder(),
            tls,
            mutual: false,
            layer,
        });
    }
    let ports = listeners.iter().map(|bound| bound.addr.port()).collect();
    (ports, following(&listeners))
}

#[tokio::test]
async fn a_replica_follows_at_its_replication_listener_that_takes_a_name_in_the_clear() {
    let gateway: Door<'_> = (&[Scope::Gateway, Scope::Control], Layer::AsTold, false);
    let unnamed: Door<'_> = (&[Scope::Replication], Layer::Off, false);
    let sealed: Door<'_> = (&[Scope::Replication], Layer::AsTold, true);
    let open: Door<'_> = (&[Scope::Control, Scope::Replication], Layer::AsTold, false);
    // Each reason, by the replication listener that gives it.
    assert_eq!(followed(&[]).await.1, Err(NoFollow::Listener));
    assert_eq!(followed(&[gateway]).await.1, Err(NoFollow::Listener));
    assert_eq!(
        followed(&[gateway, unnamed]).await.1,
        Err(NoFollow::Principal)
    );
    assert_eq!(followed(&[gateway, sealed]).await.1, Err(NoFollow::Tls));
    // The listener a standby can dial, past those it cannot.
    let (ports, port) = followed(&[gateway, unnamed, sealed, open]).await;
    assert_eq!(port.ok(), ports.last().copied());
}

#[tokio::test]
async fn a_replica_follows_under_a_shared_lease_alone_and_is_synced_within_its_poll_interval() {
    // A lease of its own: nobody to follow, whatever its listeners.
    let mut alone = Controller::bind(valid()).await.expect("the listeners bind");
    alone
        .identity_as_told("replication")
        .expect("a listener of the configuration");
    assert_eq!(alone.follows(), Err(NoFollow::Lease));
    // A shared lease: the replication listener's port, once it takes
    // its callers' names.
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut config = valid();
    config.election.lease = election::LeasePlace::File(dir.path().join("lease"));
    config.load_poll_interval = Duration::from_secs(30);
    let mut replica = Controller::bind(config).await.expect("the listeners bind");
    assert_eq!(replica.follows(), Err(NoFollow::Principal));
    replica
        .identity_as_told("replication")
        .expect("a listener of the configuration");
    let port = replica
        .addresses()
        .of(Scope::Replication)
        .map(|addr| addr.port());
    assert_eq!(replica.follows().ok(), port);
    // Synced reads the configured poll interval, not the stock one.
    assert_eq!(replica.shared.lag.within(), Duration::from_secs(30));
}

/// The bind builds the document door under the replica's fence wherever
/// another replica can take the lease, with no spec source too. Until
/// that replica leads its door reads no lead, so the door's hook
/// settles no worker's status into the table of a replica that
/// follows, and a file's start there passes its document and applies
/// none. A replica alone over its lease writes always, and its file's
/// start applies.
#[tokio::test]
async fn the_bind_puts_the_door_under_the_fence_of_a_shared_lease() {
    let alone = Controller::bind(valid()).await.expect("the listeners bind");
    assert!(alone.shared.generations.source_writes());
    assert!(!alone.shared.generations.starts_deferred());

    let dir = tempfile::tempdir().expect("a temp dir");
    let mut config = valid();
    config.election.lease = election::LeasePlace::File(dir.path().join("lease"));
    let replica = Controller::bind(config).await.expect("the listeners bind");
    let fence = replica.elector.fence();
    assert_eq!(fence.leader_term(Instant::now()), None, "it has not led");
    assert!(!replica.shared.generations.source_writes());
    assert!(replica.shared.generations.starts_deferred());
}

#[tokio::test]
async fn a_replica_that_follows_is_refused_port_zero_before_it_serves() {
    // The replication listener on port 0, by its name.
    let mut config = valid();
    assert_eq!(unnamed_port(&config.listeners), Some("replication"));
    // A lease of its own: nobody dials this replica as a leader.
    let mut alone = Controller::bind(valid()).await.expect("the listeners bind");
    alone
        .identity_as_told("replication")
        .expect("a listener of the configuration");
    assert_eq!(alone.names_its_port(), Ok(()));
    // A shared lease: the replica follows once the listener takes its
    // callers' names, and is refused there.
    let dir = tempfile::tempdir().expect("a temp dir");
    config.election.lease = election::LeasePlace::File(dir.path().join("lease"));
    let mut replica = Controller::bind(config.clone())
        .await
        .expect("the listeners bind");
    assert_eq!(replica.names_its_port(), Ok(()));
    replica
        .identity_as_told("replication")
        .expect("a listener of the configuration");
    let refused = ConfigError::ReplicationPortUnnamed("replication".to_owned());
    assert_eq!(replica.names_its_port(), Err(refused.clone()));
    // Its serve ends by that refusal, though no shutdown came.
    match timeout(Duration::from_secs(10), replica.serve(pending())).await {
        Ok(Err(Error::Config(told))) => assert_eq!(told, refused),
        other => panic!("the refusal, not {other:?}"),
    }
    // A named port is kept.
    for listener in &mut config.listeners {
        listener.addr.set_port(30_100);
    }
    assert_eq!(unnamed_port(&config.listeners), None);
}

#[tokio::test]
async fn a_standbys_word_follows_its_sync_with_no_step_of_the_election() {
    use std::{convert::Infallible, future::Future};

    use tonic_health::{
        pb::{
            health_check_response::ServingStatus as Word, health_server::Health as _,
            HealthCheckRequest,
        },
        server::{health_reporter, HealthService},
    };

    use crate::store::MemoryStore;

    /// Runs the teller until the overall word is `wanted`; fails after
    /// ten seconds.
    async fn told(
        teller: &mut (impl Future<Output = Infallible> + Unpin),
        service: &HealthService,
        wanted: Word,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let overall = HealthCheckRequest {
                service: String::new(),
            };
            let answer = service.check(tonic::Request::new(overall)).await;
            let word = answer.expect("the overall name is known").into_inner();
            if word.status() == wanted {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the word is not {wanted:?} after ten seconds"
            );
            tokio::select! {
                never = &mut *teller => match never {},
                () = tokio::time::sleep(Duration::from_millis(5)) => {}
            }
        }
    }

    let (elector, _phase, _steps, _) = fenced();
    // A standby under a bound short enough to wait out, and long
    // beside a slow host's pauses.
    let bound = Duration::from_secs(1);
    let lag = Arc::new(Lag::new(bound));
    lag.follow();
    let (_standing, stepped) = watch::channel(ElectionObservation::default());
    let (phase, role) = detached(elector.fence(), stepped, Arc::clone(&lag));
    phase.send_replace(Phase::Serving);
    let (reporter, _server) = health_reporter();
    let service = HealthService::from_health_reporter(reporter.clone());
    let store = MemoryStore::new();
    let reporters = [reporter];
    let teller = tell_readiness(role, &reporters, None, true, "controller-0", &store);
    tokio::pin!(teller);

    // No frame yet: the word is told NOT_SERVING.
    told(&mut teller, &service, Word::NotServing).await;
    // A frame: the lag tells the turn and the word follows.
    let first = Instant::now();
    lag.applied(first);
    told(&mut teller, &service, Word::Serving).await;
    // No further frame: the word lapses with the sync by the clock
    // alone, with no step of the election and no move of the run loop.
    told(&mut teller, &service, Word::NotServing).await;
    // An apply that straddled the lapse: its frame is dated inside the
    // bound and written after the word turned. The word follows it.
    lag.applied(first + bound - Duration::from_millis(1));
    told(&mut teller, &service, Word::Serving).await;
}
