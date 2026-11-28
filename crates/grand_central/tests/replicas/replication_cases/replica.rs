//! A pair of replicas over one lease, both served in the case's own process:
//! what a standby holds while it follows, what its probes say, and what it
//! serves once it leads.

use std::{
    net::{IpAddr, SocketAddr},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use smg_grand_central::{
    engine::{DemandFacts, WireModes},
    service::{
        controller::{Config, Controller, LeasePlace, Listener, Readiness, Scope, State},
        proto::{self, replication_event::Event as Sent},
    },
    store::{EncodeHandoff, EncodeOffer, MemoryStore, Registrant, WorkerFacts},
    worker_host::{Members, Objects},
};
use tokio::{sync::oneshot, task::JoinSet};
use tonic::{transport::Channel, Code};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

use crate::{
    bind_drawn, churn, client, facts, lays, net::http_get, place, replicate, settings, shared,
    snapshot, turn, writes, written_past, Hands, Twin, FOLLOWER, LEADER,
};

impl Hands for Replica {
    fn workers(&self) -> &Arc<Objects<Members>> {
        &self.workers
    }

    fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }
}

/// One replica of a pair over one lease, as a case reaches it.
struct Replica {
    replication: SocketAddr,
    admin: SocketAddr,
    store: Arc<MemoryStore>,
    /// The holder of the replica's worker objects: what a worker the case
    /// registers is written through.
    workers: Arc<Objects<Members>>,
    readiness: Readiness,
    stop: oneshot::Sender<()>,
    served: JoinSet<()>,
}

impl Replica {
    /// Binds a replica of the lease at `place` on `ip`, its replication
    /// listener at `port`, and serves it. `0` takes a number of the pairs'
    /// own ([`pair::ports`]), the next one free on `ip` and, for a pair's
    /// leader, on its standby's address too ([`pair::free`]): the number a
    /// standby of this replica then binds on its own address.
    async fn serve(id: &str, ip: IpAddr, port: u16, place: LeasePlace) -> Self {
        let listener = |name: &str, scopes: &[Scope], port| {
            Listener::of(
                name.to_owned(),
                SocketAddr::new(ip, port),
                scopes.to_vec(),
                None,
            )
        };
        let config = |port| {
            let mut configuration = Config::of(
                id.to_owned(),
                vec![
                    listener("gateway", &[Scope::Gateway], 0),
                    listener("control", &[Scope::Control], 0),
                    listener("replication", &[Scope::Replication], port),
                ],
                SocketAddr::new(ip, 0),
            );
            configuration.shutdown_settle = Duration::from_millis(50);
            configuration.shutdown_grace = Duration::from_secs(5);
            configuration.decide_shards = 2;
            configuration.election.lease = place.clone();
            configuration.election.settings = settings();
            configuration.election.advertise = None;
            configuration
        };
        let mut controller = if port == 0 {
            bind_drawn(ip, (ip == LEADER).then_some(FOLLOWER), config).await
        } else {
            Controller::bind(config(port)).await.expect("bind")
        };
        for name in ["control", "replication"] {
            controller
                .identity_as_told(name)
                .expect("a listener of the configuration");
        }
        let replication = controller
            .addresses()
            .of(Scope::Replication)
            .expect("the scope is mounted");
        let admin = controller.addresses().admin;
        let (store, readiness) = (controller.store(), controller.readiness());
        let workers = controller.workers();
        let (stop, stopped) = oneshot::channel::<()>();
        let mut served = JoinSet::new();
        served.spawn(async move {
            let shutdown = async {
                let _ = stopped.await;
            };
            controller.serve(shutdown).await.expect("a clean drain");
        });
        Self {
            replication,
            admin,
            store,
            workers,
            readiness,
            stop,
            served,
        }
    }

    /// Resolves once the replica leads, within 30 s.
    async fn leads(&mut self) {
        let led = self.readiness.wait_for(State::Leader);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), led).await,
            Ok(Some(State::Leader))
        );
    }

    /// Resolves once the replica's state is `state`, within 30 s.
    async fn reads(&mut self, state: State) {
        let read = self.readiness.wait_for(state);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), read).await,
            Ok(Some(state))
        );
    }

    /// The planned stop, which releases the lease, and its drain.
    async fn stop(mut self) {
        self.stop.send(()).expect("the serve still runs");
        let drained = tokio::time::timeout(Duration::from_secs(30), self.served.join_next()).await;
        assert!(
            matches!(drained, Ok(Some(Ok(())))),
            "a clean drain within 30 s"
        );
    }

    /// What the replica holds, without its instants: its place, then its
    /// epoch and its rows.
    fn held(&self) -> ((u64, u64), impl PartialEq + std::fmt::Debug) {
        let cut = self.workers.checkpoint(&self.store);
        let rows: Vec<_> = cut
            .rows
            .into_iter()
            .map(|row| (row.id, row.facts, row.status, row.revision))
            .collect();
        ((cut.at.seq, cut.at.term), (cut.epoch, rows))
    }

    /// Waits, within 20 s, until the replica stands at `place`.
    async fn stands_at(&self, place: (u64, u64)) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while self.held().0 != place {
            assert!(
                Instant::now() < deadline,
                "the replica stands at {:?}, not at {place:?}, after 20 s",
                self.held().0
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// How many polls the replica whose admin listener is `admin` has sent, by
/// its metrics page.
async fn polls(admin: SocketAddr) -> u64 {
    let (status, page) = http_get(admin, "/metrics").await.expect("the metrics page");
    assert_eq!(status, 200);
    let counter = "smg_grand_central_load_polls_total{mode=\"poll\"} ";
    let count = page.lines().find_map(|line| line.strip_prefix(counter));
    count
        .and_then(|count| count.trim().parse().ok())
        .expect("the polls' counter")
}

/// The `Leader` call's answer at `addr`.
async fn leader_info(addr: SocketAddr) -> proto::LeaderInfo {
    client(addr)
        .await
        .leader(proto::LeaderRequest::default())
        .await
        .expect("every listener answers")
        .into_inner()
}

/// The snapshot of a replica that took the lease, within 20 s: between its
/// take and the take's own write the call is `UNAVAILABLE`.
async fn promoted(addr: SocketAddr) -> proto::Snapshot {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        match snapshot(addr).await {
            Ok(served) => return served,
            Err(wait) => {
                assert_eq!(wait.code(), Code::Unavailable, "no snapshot: {wait}");
                assert!(Instant::now() < deadline, "no snapshot in 20 s: {wait}");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_promoted_standby_holds_every_row_of_its_leader() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let began = leader.held().0;
    let port = leader.replication.port();
    let mut standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    // A ready worker the load monitor polls, at an address nobody answers.
    // It states what a worker states for a request's hard demands, it
    // takes images, and it states an end of the encode handoff, a priority
    // and a drain window: the log's row tells the five, so the standby
    // holds the row whole. The offer's connector holds a blank: a holder's
    // write takes any text, the log tells it, and a standby reads it as
    // told.
    let mut stated = DemandFacts::default();
    stated.runtime = 7;
    stated.references = true;
    stated.host_pipeline = true;
    stated.window = 32_768;
    stated.decoding = 3;
    let mut offered = EncodeOffer::default();
    offered.handoff = EncodeHandoff::Push;
    offered.producer = true;
    offered.connector = Some("a b".to_owned());
    let polled = WorkerFacts {
        wire_modes: WireModes::GRPC,
        demand_facts: stated,
        vision: true,
        encode: offered,
        priority: -5,
        drain_settle_secs: Some(45),
        ..facts("grpc://127.0.0.1:1", 1, 0)
    };
    let written = leader
        .workers
        .upsert(&leader.store, polled, Registrant::Control);
    let polled = written.expect("the store took the registration").id();
    let revision = leader.workers.read_row(polled).expect("the row").revision;
    leader.workers.connect_row(&leader.store, polled, revision);
    // A row nobody polls, whose offer states a connector past a name's 256
    // bytes, and one that states no offer.
    let mut taking = EncodeOffer::default();
    taking.handoff = EncodeHandoff::Push;
    taking.consumer = true;
    taking.connector = Some("c".repeat(300));
    let wide = WorkerFacts {
        encode: taking,
        ..facts("http://wide:8000", 1, 0)
    };
    lays(&leader, wide);
    writes(&leader, 0);
    // The leader writes while the standby follows.
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let hands = (Arc::clone(&leader.workers), Arc::clone(&leader.store));
        let stop = Arc::clone(&stop);
        thread::spawn(move || churn(&hands, &stop))
    };
    // The events the case counts are waited for, never timed.
    written_past(&leader, began.0 + 100).await;
    stop.store(true, Ordering::Relaxed);
    let rounds = writer.join().expect("the writer");
    let (head, wrote) = leader.held();
    let events = head.0 - began.0;
    assert!(events > 100, "{rounds} rounds wrote {events} events");
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote, "{rounds} rounds");
    // The leader's load monitor polls the ready worker. The standby holds
    // the same row and polls nobody while it follows: a while later, its
    // counter still reads none.
    let deadline = Instant::now() + Duration::from_secs(20);
    while polls(leader.admin).await == 0 {
        assert!(Instant::now() < deadline, "the leader polls its worker");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    assert_eq!(polls(standby.admin).await, 0, "a follower polls nobody");
    // The standby says how far behind it is; the leader follows nobody.
    let info = leader_info(standby.replication).await;
    assert!(!info.is_leader && info.synced, "{info:?}");
    let lag = info.sync.map(|sync| sync.replication_lag_ms);
    assert!(lag.is_some_and(|lag| lag < 5_000), "{lag:?}");
    let own = leader_info(leader.replication).await;
    assert!(
        own.is_leader && !own.synced && own.sync.is_none(),
        "{own:?}"
    );

    // The leader stops: the standby takes the lease with what it held and
    // serves it under a term of its own.
    leader.stop().await;
    standby.leads().await;
    let served = promoted(standby.replication).await;
    assert_eq!(
        served.replication_seq,
        head.0 + 1,
        "its term starts after the last event"
    );
    assert!(served.replication_term > head.1);
    let (place, holds) = standby.held();
    assert_eq!(place, (served.replication_seq, served.replication_term));
    assert_eq!(holds, wrote);
    // It reads its engines itself from its take: its own load monitor
    // polls the ready worker.
    let deadline = Instant::now() + Duration::from_secs(20);
    while polls(standby.admin).await == 0 {
        assert!(
            Instant::now() < deadline,
            "the promoted standby polls its worker"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // A leader follows nobody: its answer says no sync state.
    let promoted = leader_info(standby.replication).await;
    assert!(
        promoted.is_leader && !promoted.synced && promoted.sync.is_none(),
        "{promoted:?}"
    );
    standby.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_standby_that_joins_late_is_restored_whole_from_a_snapshot_past_one_message() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let began = leader.held().0;
    // More status changes than the log retains of the watched kinds
    // (4,096): the place a new standby names is gone from it.
    let id = lays(&leader, facts("http://late:8000", 1, 0));
    for round in 0..4_300 {
        turn(&leader, id, round);
    }
    // Rows of a few kilobytes each, by their key: more than one message of
    // the transport holds, so the snapshot is several parts.
    let wide = "k".repeat(2_000);
    for row in 0..1_500 {
        lays(&leader, facts(&format!("http://{wide}-{row}:8000"), 1, 0));
    }
    writes(&leader, 3);
    let port = leader.replication.port();
    let standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    let (head, wrote) = leader.held();
    assert!(head.0 - began.0 > 4_300);
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote);
    // It follows the log from the snapshot's place.
    lays(&leader, facts("http://after:8000", 1, 0));
    let (head, wrote) = leader.held();
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote);
    standby.stop().await;
    leader.stop().await;
}

/// A promoted standby holds its leader's history from the first write: a
/// reader that holds nothing is fed it whole, then the new term's start. A
/// standby restored from a snapshot tells such a reader to start again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_promoted_standby_feeds_a_reader_that_holds_nothing_from_its_leaders_first_write() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let port = leader.replication.port();
    let mut standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    writes(&leader, 3);
    let (head, wrote) = leader.held();
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote);
    leader.stop().await;
    standby.leads().await;
    let served = promoted(standby.replication).await;
    let mut reader = replicate(standby.replication, place(0, 0))
        .await
        .expect("a stream");
    let fed = reader.until(served.replication_seq).await;
    let mut twin = Twin::default();
    for event in &fed {
        twin.apply(event).expect("an event the leader journals");
    }
    let apart = twin.apart(&Twin::of(&served));
    assert!(apart.is_empty(), "{apart:?}");
    // The first leader's term's start and its writes, then the promoted
    // replica's own term's start.
    let starts: Vec<_> = fed
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event.event, Some(Sent::TermStart(_))))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(starts, [0, fed.len() - 1], "{} events", fed.len());
    standby.stop().await;
}

/// The listener's overall word: the health service's answer for no name.
async fn overall(addr: SocketAddr) -> ServingStatus {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("a uri")
        .connect()
        .await
        .expect("the listener accepts");
    HealthClient::new(channel)
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("the health service answers for no name")
        .into_inner()
        .status()
}

/// The sample of `series`, a series of no label, on the replica's metrics
/// page.
async fn sample(admin: SocketAddr, series: &str) -> String {
    let (status, page) = http_get(admin, "/metrics").await.expect("the metrics page");
    assert_eq!(status, 200);
    let sample = page
        .lines()
        .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '));
    sample.expect("the series").to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_standby_passes_its_probes_only_while_it_is_synced_with_its_leader() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let port = leader.replication.port();

    // A standby that cannot reach its leader: it dials the leader's
    // address at its own replication port, where no replication listener
    // answers. It serves, and no probe of it passes.
    let mut apart = Replica::serve("replication-1", FOLLOWER, 0, shared(&dir)).await;
    assert_ne!(apart.replication.port(), port);
    apart.reads(State::Syncing).await;
    let (status, body) = http_get(apart.admin, "/readiness")
        .await
        .expect("the probe");
    assert_eq!(
        (status, body.as_str()),
        (503, r#"{"status":"not ready","reason":"syncing"}"#)
    );
    assert_eq!(overall(apart.replication).await, ServingStatus::NotServing);
    let state = "smg_grand_central_state{state=\"syncing\"}";
    assert_eq!(sample(apart.admin, state).await, "1");
    assert_eq!(
        sample(apart.admin, "smg_grand_central_replication_synced").await,
        "0"
    );
    apart.stop().await;

    // One that follows its leader passes them once its first frame is
    // applied, and says so on its metrics page.
    let mut standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    standby.reads(State::Follower).await;
    let (status, body) = http_get(standby.admin, "/readiness")
        .await
        .expect("the probe");
    assert_eq!(
        (status, body.as_str()),
        (200, r#"{"status":"ready","role":"follower"}"#)
    );
    // The word is told after the state turns.
    let deadline = Instant::now() + Duration::from_secs(20);
    while overall(standby.replication).await != ServingStatus::Serving {
        assert!(
            Instant::now() < deadline,
            "a synced standby's word is not SERVING after 20 s"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        sample(standby.admin, "smg_grand_central_replication_synced").await,
        "1"
    );
    assert_eq!(sample(standby.admin, state).await, "0");
    standby.stop().await;
    leader.stop().await;
}

/// A row its leader wrote reaches the published world of a standby that
/// still follows. The standby's applier leaves the followed row's build
/// owed, and no write of the standby's own comes behind it: its own serve
/// pays the publication, as a leader's serve pays its own. The row is on
/// file with its build owed from the applier's one hold of the table, so a
/// read that finds the row and no build owed found the build paid.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_followed_row_reaches_the_published_world_of_a_standby() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let port = leader.replication.port();
    let mut standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    standby.reads(State::Follower).await;
    let published = |replica: &Replica| {
        // The row the holder reads under the key, where the holder's cut
        // holds it under that key: the rows the holder and the store both
        // hold.
        let row = replica.workers.read_row_under("http://a:8000");
        let cut = replica.workers.checkpoint(&replica.store).rows;
        let on_file = row.is_some_and(|row| {
            cut.iter()
                .any(|told| told.id == row.id && told.facts.key == "http://a:8000")
        });
        on_file && replica.store.publish_due().is_none()
    };
    assert!(!published(&standby), "no row is written yet");

    // One row on the leader, and no write behind it on either replica.
    lays(&leader, facts("http://a:8000", 1, 0));
    let deadline = Instant::now() + Duration::from_secs(10);
    while !published(&standby) {
        assert!(
            Instant::now() < deadline,
            "the standby published no followed row within 10 s"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // It published as a follower: the lead did not turn.
    standby.reads(State::Follower).await;
    standby.stop().await;
    leader.stop().await;
}
